# Copyright (c) OpenMMLab. All rights reserved.
import copy
import re
import warnings
from typing import Dict, Optional, Tuple, Union
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
from mmengine.runner.amp import autocast
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType
from ..layers import SinePositionalEncoding
from ..layers.transformer.grounding_dino_layers import (
    GroundingDinoTransformerDecoder, GroundingDinoTransformerEncoder)
from .dino import DINO
from .glip import (create_positive_map, create_positive_map_label_to_token,
                   run_ner)


def clean_label_name(name: str) -> str:
    name = re.sub(r'\(.*\)', '', name)
    name = re.sub(r'_', ' ', name)
    name = re.sub(r'  ', ' ', name)
    return name


def chunks(lst: list, n: int) -> list:
    """Yield successive n-sized chunks from lst."""
    all_ = []
    for i in range(0, len(lst), n):
        data_index = lst[i:i + n]
        all_.append(data_index)
    counter = 0
    for i in all_:
        counter += len(i)
    assert (counter == len(lst))

    return all_

class PromptLearner(nn.Module):
    def __init__(self, prompt_type, n_ctx, text_embedding_dim, visual_embedding_dim):

        super().__init__()
        self.prompt_type = prompt_type
        self.n_ctx = n_ctx

        # initialize prompt 
        if self.prompt_type == 'glip_style':
            # GLIP-style prompt-tuning (after text encoder, BERT, inspiration: GLIP maskrcnn_benchmark/modeling/rpn/vldyhead.py)
            self.ctx = torch.nn.Linear(text_embedding_dim, 1000, bias=False)
            self.ctx.weight.data.fill_(0.0)

        elif self.prompt_type == 'fusion_text':
            # add prompt after the encoder
            self.ctx = torch.nn.Linear(visual_embedding_dim, 1000, bias=False)
            self.ctx.weight.data.fill_(0.0)

        elif self.prompt_type == 'coop_new_class':
            # CoOp-style propmt-tuning (after BERT embeddings, before BERT encoder)
            # Add prompt as a new class ('Car', 'Truck', 'XXXX')
            # self.n_ctx = 32
            assert (self.n_ctx > 0)
            ctx_vectors = torch.empty(self.n_ctx, text_embedding_dim)
            nn.init.normal_(ctx_vectors, std=0.02)
            self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        elif self.prompt_type == 'coop_csc':
            # append a class-specific-context (csc) vector to each class name
            # ('XXXX Car', 'XXXX Truck')  -> each XXXX is assigned a separate traianble embedding
            # self.n_ctx = 2
            assert (self.n_ctx > 0)
            # support max 256 classes as of now
            ctx_vectors = torch.empty(256, self.n_ctx, text_embedding_dim)
            nn.init.normal_(ctx_vectors, std=0.02)
            self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        elif self.prompt_type == 'cocoop':
            # unified context vector added to each class name
            # ('XXXX Car', 'XXXX Truck')  -> each XXXX is assigned the same traianble embedding
            # meta-token from the meta-net is also added to the common context vector
            # initialize the context vector: this will be prepended to all class name embeddings
            # self.n_ctx = 4
            assert (self.n_ctx > 0)
            ctx_dim = text_embedding_dim
            ctx_vectors = torch.empty(self.n_ctx, ctx_dim)
            nn.init.normal_(ctx_vectors, std=0.02)
            self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
            # initialize the meta net
            vis_dim = visual_embedding_dim
            self.meta_net = nn.Sequential(OrderedDict([
                ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
                ("relu", nn.ReLU(inplace=True)),
                ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
            ]))


    def insert_dummy_caption(self, text_prompts, name_lens, language_model):
        
        if self.prompt_type == 'coop_new_class':
            # add dummy text to caption corresponding to context embeddings to be added later
            prompt_suffix = (('X') * (self.ctx.shape[0]*2),)
            text_prompts = [p + prompt_suffix for p in text_prompts]
            # verify that the number of context tokens equals the length of context vector
            # ctx_tokens = self.get_tokens_and_prompts(text_prompts[0][-1], True)[0]['input_ids'][0]
            # num_ctx_tokens = len(ctx_tokens) - 3  # subtract the SOS and EOS token, and a space [SOS,XX,XX, ,EOS]
            # assert (num_ctx_tokens == self.ctx.shape[0])

        elif self.prompt_type in ['coop_csc', 'cocoop']:
            # prepend context to each class name
            prompt_prefix = ('X') * (self.ctx.shape[-2]*2) + ' '
            for i in range(len(text_prompts)):
                text_prompts[i] = tuple([prompt_prefix + e for e in text_prompts[i]])
            # update the name lengths
            name_lens = []
            for p in text_prompts[0]:
                name_lens.append(len(language_model.tokenizer(p)['input_ids'])-2)  # subtract the SOS and EOS tokens
            
        return text_prompts, name_lens

    def get_text_embeddings(self, new_text_prompts, name_lens, visual_features, language_model):
        # for coop variants, insert context vector at the word embedding level 
        # within the language model
        if self.prompt_type == 'coop_new_class':   
            # ctx is (n_ctx, ctx_dim)         
            text_dict = language_model(new_text_prompts, self.ctx, name_lens, self.prompt_type)
        elif self.prompt_type == 'coop_csc':            
            # ctx is (256, n_ctx, ctx_dim)
            text_dict = language_model(new_text_prompts, self.ctx, name_lens, self.prompt_type)
        elif self.prompt_type == 'cocoop':
            ctx = self.ctx                          # (n_ctx, ctx_dim)
            vfeat = visual_features[-1]             # (batch, vis_dim, h, w)
            vfeat = vfeat.mean(dim=(2,3))           # (batch, vis_dim)
            vfeat = vfeat / vfeat.norm(dim=-1, keepdim=True)
            bias = self.meta_net(vfeat)             # (batch, ctx_dim)
            bias = bias.unsqueeze(1)                # (batch, 1, ctx_dim)
            ctx = ctx.unsqueeze(0)                  # (1, n_ctx, ctx_dim)
            ctx_shifted = ctx + bias                # (batch, n_ctx, ctx_dim)
            text_dict = language_model(new_text_prompts, ctx_shifted, name_lens, self.prompt_type)
        else:
            text_dict = language_model(new_text_prompts)

        # for glip-style prompt, add context vector to the text embeddings 
        # after the language model
         
        if self.prompt_type == 'glip_style':
            # tunable linear layer for prompt tuning
            embedding = text_dict['embedded']
            embedding = self.ctx.weight[:embedding.size(1), :].unsqueeze(0) + embedding
            text_dict['embedded'] = embedding
            text_dict['hidden'] = self.ctx.weight[:embedding.size(1), :].unsqueeze(0) + text_dict['hidden']

        return text_dict


@MODELS.register_module()
class GroundingDINO(DINO):
    """Implementation of `Grounding DINO: Marrying DINO with Grounded Pre-
    Training for Open-Set Object Detection.

    <https://arxiv.org/abs/2303.05499>`_

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/GroundingDINO>`_.
    """

    def __init__(self,
                 language_model,
                 *args,
                 use_autocast=False,
                 # prompt related
                 prompt_type=None,   # prompting method
                 n_ctx = 0,     # no. of prompt context vectors
                 **kwargs) -> None:

        self.language_model_cfg = language_model
        self._special_tokens = '. '
        self.use_autocast = use_autocast
        self.prompt_type = prompt_type
        self.n_ctx = n_ctx
        
        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = GroundingDinoTransformerEncoder(**self.encoder)
        self.decoder = GroundingDinoTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

        # text modules
        self.language_model = MODELS.build(self.language_model_cfg)

        # initialize prompt 
        self.prompt_learner = PromptLearner(self.prompt_type, self.n_ctx, 
                                            text_embedding_dim=self.language_model.language_backbone.body.language_dim, 
                                            visual_embedding_dim=self.embed_dims)

        self.text_feat_map = nn.Linear(
            self.language_model.language_backbone.body.language_dim,
            self.embed_dims,
            bias=True)


    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super().init_weights()
        nn.init.constant_(self.text_feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.text_feat_map.weight.data)

    def to_enhance_text_prompts(self, original_caption, enhanced_text_prompts):
        caption_string = ''
        tokens_positive = []
        for idx, word in enumerate(original_caption):
            if word in enhanced_text_prompts:
                enhanced_text_dict = enhanced_text_prompts[word]
                if 'prefix' in enhanced_text_dict:
                    caption_string += enhanced_text_dict['prefix']
                start_i = len(caption_string)
                if 'name' in enhanced_text_dict:
                    caption_string += enhanced_text_dict['name']
                else:
                    caption_string += word
                end_i = len(caption_string)
                tokens_positive.append([[start_i, end_i]])

                if 'suffix' in enhanced_text_dict:
                    caption_string += enhanced_text_dict['suffix']
            else:
                tokens_positive.append(
                    [[len(caption_string),
                      len(caption_string) + len(word)]])
                caption_string += word
            caption_string += self._special_tokens
        return caption_string, tokens_positive

    def to_plain_text_prompts(self, original_caption):
        caption_string = ''
        tokens_positive = []
        for idx, word in enumerate(original_caption):
            tokens_positive.append(
                [[len(caption_string),
                  len(caption_string) + len(word)]])
            caption_string += word
            caption_string += self._special_tokens
        return caption_string, tokens_positive

    def get_tokens_and_prompts(
        self,
        original_caption: Union[str, list, tuple],
        custom_entities: bool = False,
        enhanced_text_prompts: Optional[ConfigType] = None
    ) -> Tuple[dict, str, list]:
        """Get the tokens positive and prompts for the caption."""
        if isinstance(original_caption, (list, tuple)) or custom_entities:
            if custom_entities and isinstance(original_caption, str):
                original_caption = original_caption.strip(self._special_tokens)
                original_caption = original_caption.split(self._special_tokens)
                original_caption = list(
                    filter(lambda x: len(x) > 0, original_caption))

            original_caption = [clean_label_name(i) for i in original_caption]

            if custom_entities and enhanced_text_prompts is not None:
                caption_string, tokens_positive = self.to_enhance_text_prompts(
                    original_caption, enhanced_text_prompts)
            else:
                caption_string, tokens_positive = self.to_plain_text_prompts(
                    original_caption)

            # NOTE: Tokenizer in Grounding DINO is different from
            # that in GLIP. The tokenizer in GLIP will pad the
            # caption_string to max_length, while the tokenizer
            # in Grounding DINO will not.
            tokenized = self.language_model.tokenizer(
                [caption_string],
                padding='max_length'
                if self.language_model.pad_to_max else 'longest',
                return_tensors='pt')
            entities = original_caption
        else:
            if not original_caption.endswith('.'):
                original_caption = original_caption + self._special_tokens
            # NOTE: Tokenizer in Grounding DINO is different from
            # that in GLIP. The tokenizer in GLIP will pad the
            # caption_string to max_length, while the tokenizer
            # in Grounding DINO will not.
            tokenized = self.language_model.tokenizer(
                [original_caption],
                padding='max_length'
                if self.language_model.pad_to_max else 'longest',
                return_tensors='pt')
            tokens_positive, noun_phrases = run_ner(original_caption)
            entities = noun_phrases
            caption_string = original_caption

        return tokenized, caption_string, tokens_positive, entities

    def get_positive_map(self, tokenized, tokens_positive):
        positive_map = create_positive_map(
            tokenized,
            tokens_positive,
            max_num_entities=self.bbox_head.cls_branches[
                self.decoder.num_layers].max_text_len)
        positive_map_label_to_token = create_positive_map_label_to_token(
            positive_map, plus=1)
        return positive_map_label_to_token, positive_map

    def get_tokens_positive_and_prompts(
        self,
        original_caption: Union[str, list, tuple],
        custom_entities: bool = False,
        enhanced_text_prompt: Optional[ConfigType] = None,
        tokens_positive: Optional[list] = None,
    ) -> Tuple[dict, str, Tensor, list]:
        """Get the tokens positive and prompts for the caption.

        Args:
            original_caption (str): The original caption, e.g. 'bench . car .'
            custom_entities (bool, optional): Whether to use custom entities.
                If ``True``, the ``original_caption`` should be a list of
                strings, each of which is a word. Defaults to False.

        Returns:
            Tuple[dict, str, dict, str]: The dict is a mapping from each entity
            id, which is numbered from 1, to its positive token id.
            The str represents the prompts.
        """
        if tokens_positive is not None:
            if tokens_positive == -1:
                if not original_caption.endswith('.'):
                    original_caption = original_caption + self._special_tokens
                return None, original_caption, None, original_caption
            else:
                if not original_caption.endswith('.'):
                    original_caption = original_caption + self._special_tokens
                tokenized = self.language_model.tokenizer(
                    [original_caption],
                    padding='max_length'
                    if self.language_model.pad_to_max else 'longest',
                    return_tensors='pt')
                positive_map_label_to_token, positive_map = \
                    self.get_positive_map(tokenized, tokens_positive)

                entities = []
                for token_positive in tokens_positive:
                    instance_entities = []
                    for t in token_positive:
                        instance_entities.append(original_caption[t[0]:t[1]])
                    entities.append(' / '.join(instance_entities))
                return positive_map_label_to_token, original_caption, \
                    positive_map, entities

        chunked_size = self.test_cfg.get('chunked_size', -1)
        if not self.training and chunked_size > 0:
            assert isinstance(original_caption,
                              (list, tuple)) or custom_entities is True
            all_output = self.get_tokens_positive_and_prompts_chunked(
                original_caption, enhanced_text_prompt)
            positive_map_label_to_token, \
                caption_string, \
                positive_map, \
                entities = all_output
        else:
            tokenized, caption_string, tokens_positive, entities = \
                self.get_tokens_and_prompts(
                    original_caption, custom_entities, enhanced_text_prompt)
            positive_map_label_to_token, positive_map = self.get_positive_map(
                tokenized, tokens_positive)
        return positive_map_label_to_token, caption_string, \
            positive_map, entities

    def get_tokens_positive_and_prompts_chunked(
            self,
            original_caption: Union[list, tuple],
            enhanced_text_prompts: Optional[ConfigType] = None):
        chunked_size = self.test_cfg.get('chunked_size', -1)
        original_caption = [clean_label_name(i) for i in original_caption]

        original_caption_chunked = chunks(original_caption, chunked_size)
        ids_chunked = chunks(
            list(range(1,
                       len(original_caption) + 1)), chunked_size)

        positive_map_label_to_token_chunked = []
        caption_string_chunked = []
        positive_map_chunked = []
        entities_chunked = []

        for i in range(len(ids_chunked)):
            if enhanced_text_prompts is not None:
                caption_string, tokens_positive = self.to_enhance_text_prompts(
                    original_caption_chunked[i], enhanced_text_prompts)
            else:
                caption_string, tokens_positive = self.to_plain_text_prompts(
                    original_caption_chunked[i])
            tokenized = self.language_model.tokenizer([caption_string],
                                                      return_tensors='pt')
            if tokenized.input_ids.shape[1] > self.language_model.max_tokens:
                warnings.warn('Inputting a text that is too long will result '
                              'in poor prediction performance. '
                              'Please reduce the --chunked-size.')
            positive_map_label_to_token, positive_map = self.get_positive_map(
                tokenized, tokens_positive)

            caption_string_chunked.append(caption_string)
            positive_map_label_to_token_chunked.append(
                positive_map_label_to_token)
            positive_map_chunked.append(positive_map)
            entities_chunked.append(original_caption_chunked[i])

        return positive_map_label_to_token_chunked, \
            caption_string_chunked, \
            positive_map_chunked, \
            entities_chunked

    def forward_transformer(
        self,
        img_feats: Tuple[Tensor],
        text_dict: Dict,
        batch_data_samples: OptSampleList = None,
    ) -> Dict:
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)

        encoder_outputs_dict = self.forward_encoder(
            **encoder_inputs_dict, text_dict=text_dict)

        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        return head_inputs_dict

    def forward_encoder(self, feat: Tensor, feat_mask: Tensor,
                        feat_pos: Tensor, spatial_shapes: Tensor,
                        level_start_index: Tensor, valid_ratios: Tensor,
                        text_dict: Dict) -> Dict:
        text_token_mask = text_dict['text_token_mask']
        memory, memory_text = self.encoder(
            query=feat,
            query_pos=feat_pos,
            key_padding_mask=feat_mask,  # for self_attn
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            # for text encoder
            memory_text=text_dict['embedded'],
            text_attention_mask=~text_token_mask,
            position_ids=text_dict['position_ids'],
            text_self_attention_masks=text_dict['masks'])
        
        if self.prompt_type == 'fusion_text':
            embedding = memory_text
            embedding = self.prompt_learner.ctx.weight[:embedding.size(1), :].unsqueeze(0) + embedding
            memory_text = embedding
            # text_dict['hidden'] = self.tunable_linear.weight[:embedding.size(1), :].unsqueeze(0) + text_dict['hidden']

        encoder_outputs_dict = dict(
            memory=memory,
            memory_mask=feat_mask,
            spatial_shapes=spatial_shapes,
            memory_text=memory_text,
            text_token_mask=text_token_mask)
        return encoder_outputs_dict

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        memory_text: Tensor,
        text_token_mask: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        bs, _, c = memory.shape

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)

        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers](output_memory, memory_text,
                                     text_token_mask)
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers].max_text_len
        enc_outputs_coord_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers](output_memory) + output_proposals

        # NOTE The DINO selects top-k proposals according to scores of
        # multi-class classification, while DeformDETR, where the input
        # is `enc_outputs_class[..., 0]` selects according to scores of
        # binary classification.
        topk_indices = torch.topk(
            enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1)[1]

        topk_score = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()

        query = self.query_embedding.weight[:, None, :]
        query = query.repeat(1, bs, 1).transpose(0, 1)
        if self.training:
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                self.dn_query_generator(batch_data_samples)
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact],
                                         dim=1)
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None
        reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask,
            memory_text=memory_text,
            text_attention_mask=~text_token_mask,
        )
        # NOTE DINO calculates encoder losses on scores and coordinates
        # of selected top-k encoder queries, while DeformDETR is of all
        # encoder queries.
        head_inputs_dict = dict(
            enc_outputs_class=topk_score,
            enc_outputs_coord=topk_coords,
            dn_meta=dn_meta) if self.training else dict()
        # append text_feats to head_inputs_dict
        head_inputs_dict['memory_text'] = memory_text
        head_inputs_dict['text_token_mask'] = text_token_mask
        return decoder_inputs_dict, head_inputs_dict

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        

        # extract visual features
        if self.use_autocast:
            with autocast(enabled=True):
                visual_features = self.extract_feat(batch_inputs)
        else:
            visual_features = self.extract_feat(batch_inputs)

        # construct text prompts (concatenation of class names)
        text_prompts = [
            data_samples.text for data_samples in batch_data_samples
        ]

        # compute number of tokens for each class name (assuming all text prompts for the batch are same)
        name_lens = []
        for p in text_prompts[0]:
            name_lens.append(len(self.language_model.tokenizer(p)['input_ids'])-2)  # subtract the SOS and EOS tokens

        # insert dummy text caption for CoOp variants
        text_prompts, name_lens = self.prompt_learner.insert_dummy_caption(text_prompts, name_lens, self.language_model)

        gt_labels = [
            data_samples.gt_instances.labels
            for data_samples in batch_data_samples
        ]

        # tokenize prompts and compute positive maps
        if 'tokens_positive' in batch_data_samples[0]:
            # TODO: handle this part of the if-else statement
            tokens_positive = [
                data_samples.tokens_positive
                for data_samples in batch_data_samples
            ]
            positive_maps = []
            for token_positive, text_prompt, gt_label in zip(
                    tokens_positive, text_prompts, gt_labels):
                tokenized = self.language_model.tokenizer(
                    [text_prompt],
                    padding='max_length'
                    if self.language_model.pad_to_max else 'longest',
                    return_tensors='pt')
                new_tokens_positive = [
                    token_positive[label.item()] for label in gt_label
                ]
                _, positive_map = self.get_positive_map(
                    tokenized, new_tokens_positive)
                positive_maps.append(positive_map)
            new_text_prompts = text_prompts
        else:
            new_text_prompts = []
            positive_maps = []
            if len(set(text_prompts)) == 1:
                # All the text prompts are the same,
                # so there is no need to calculate them multiple times.
                tokenized, caption_string, tokens_positive, _ = \
                    self.get_tokens_and_prompts(
                        text_prompts[0], True)
                new_text_prompts = [caption_string] * len(batch_inputs)
                for gt_label in gt_labels:
                    new_tokens_positive = [
                        tokens_positive[label] for label in gt_label
                    ]
                    _, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive)
                    positive_maps.append(positive_map)
            else:
                for text_prompt, gt_label in zip(text_prompts, gt_labels):
                    tokenized, caption_string, tokens_positive, _ = \
                        self.get_tokens_and_prompts(
                            text_prompt, True)
                    new_tokens_positive = [
                        tokens_positive[label] for label in gt_label
                    ]
                    _, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive)
                    positive_maps.append(positive_map)
                    new_text_prompts.append(caption_string)

        # extract prompt-enhanced text embeddings from the language model
        text_dict = self.prompt_learner.get_text_embeddings(new_text_prompts, name_lens, visual_features, self.language_model)

        if self.text_feat_map is not None:
            text_dict['embedded'] = self.text_feat_map(text_dict['embedded'])

        for i, data_samples in enumerate(batch_data_samples):
            positive_map = positive_maps[i].to(
                batch_inputs.device).bool().float()
            text_token_mask = text_dict['text_token_mask'][i]
            data_samples.gt_instances.positive_maps = positive_map
            data_samples.gt_instances.text_token_mask = \
                text_token_mask.unsqueeze(0).repeat(
                    len(positive_map), 1)
       
        head_inputs_dict = self.forward_transformer(visual_features, text_dict,
                                                    batch_data_samples)

        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)
        return losses

    def predict(self, batch_inputs, batch_data_samples, rescale: bool = True):
        
        # image feature extraction
        visual_feats = self.extract_feat(batch_inputs)

        # construct text prompts (concatenation of class names)
        text_prompts = []
        enhanced_text_prompts = []
        tokens_positives = []
        for data_samples in batch_data_samples:
            text_prompts.append(data_samples.text)
            if 'caption_prompt' in data_samples:
                enhanced_text_prompts.append(data_samples.caption_prompt)
            else:
                enhanced_text_prompts.append(None)
            tokens_positives.append(data_samples.get('tokens_positive', None))

        # compute number of tokens for each class name (assuming all text prompts for the batch are same)
        name_lens = []
        for p in text_prompts[0]:
            name_lens.append(len(self.language_model.tokenizer(p)['input_ids'])-2)  # subtract the SOS and EOS tokens

        # insert dummy text caption for CoOp variants
        text_prompts, name_lens = self.prompt_learner.insert_dummy_caption(text_prompts, name_lens, self.language_model)


        if 'custom_entities' in batch_data_samples[0]:
            # Assuming that the `custom_entities` flag
            # inside a batch is always the same. For single image inference
            custom_entities = batch_data_samples[0].custom_entities
        else:
            custom_entities = False

        # compute positive maps
        if len(text_prompts) == 1:
            # All the text prompts are the same,
            # so there is no need to calculate them multiple times.
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(
                    text_prompts[0], custom_entities, enhanced_text_prompts[0],
                    tokens_positives[0])
            ] * len(batch_inputs)
        else:
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(text_prompt,
                                                     custom_entities,
                                                     enhanced_text_prompt,
                                                     tokens_positive)
                for text_prompt, enhanced_text_prompt, tokens_positive in zip(
                    text_prompts, enhanced_text_prompts, tokens_positives)
            ]
        token_positive_maps, text_prompts, _, entities = zip(
            *_positive_maps_and_prompts)

        if isinstance(text_prompts[0], list):
            # chunked text prompts, only bs=1 is supported
            assert len(batch_inputs) == 1
            count = 0
            results_list = []

            entities = [[item for lst in entities[0] for item in lst]]

            for b in range(len(text_prompts[0])):
                text_prompts_once = [text_prompts[0][b]]
                token_positive_maps_once = token_positive_maps[0][b]

                # extract prompt-enhanced text embeddings from the language model
                text_dict = self.prompt_learner.get_text_embeddings(text_prompts_once, name_lens, visual_feats, self.language_model)


                # text feature map layer
                if self.text_feat_map is not None:
                    text_dict['embedded'] = self.text_feat_map(
                        text_dict['embedded'])

                batch_data_samples[
                    0].token_positive_map = token_positive_maps_once

                head_inputs_dict = self.forward_transformer(
                    copy.deepcopy(visual_feats), text_dict, batch_data_samples)
                
                # not sure if this is the right thing to do
                if self.prompt_type == 'coop_new_class':
                    skip_indices = token_positive_maps[0][(len(entities[0]))]

                    # remove ctx embeddings from text embeddings vector before passing on the bbox_head
                    head_inputs_dict['memory_text'] = torch.cat((head_inputs_dict['memory_text'][:, :skip_indices[0], :], 
                                                                head_inputs_dict['memory_text'][:, -1:, :]), dim=1)
                    head_inputs_dict['text_token_mask'] = torch.cat((head_inputs_dict['text_token_mask'][:, :skip_indices[0]], 
                                                                head_inputs_dict['text_token_mask'][:, -1:]), dim=1)
                    
                    # remove dummy class from token_positive_maps
                    token_positive_maps[0].pop(len(entities[0]))
                    for i, data_samples in enumerate(batch_data_samples):
                        data_samples.token_positive_map = token_positive_maps[i]

                pred_instances = self.bbox_head.predict(
                    **head_inputs_dict,
                    rescale=rescale,
                    batch_data_samples=batch_data_samples)[0]

                if len(pred_instances) > 0:
                    pred_instances.labels += count
                count += len(token_positive_maps_once)
                results_list.append(pred_instances)
            results_list = [results_list[0].cat(results_list)]
            is_rec_tasks = [False] * len(results_list)
        else:
            # extract prompt-enhanced text embeddings from the language model
            text_dict = self.prompt_learner.get_text_embeddings(text_prompts, name_lens, visual_feats, self.language_model)

            # text feature map layer
            if self.text_feat_map is not None:
                text_dict['embedded'] = self.text_feat_map(
                    text_dict['embedded'])

            is_rec_tasks = []
            for i, data_samples in enumerate(batch_data_samples):
                if token_positive_maps[i] is not None:
                    is_rec_tasks.append(False)
                else:
                    is_rec_tasks.append(True)
                data_samples.token_positive_map = token_positive_maps[i]

            head_inputs_dict = self.forward_transformer(
                visual_feats, text_dict, batch_data_samples)
            
            # not sure if this is the right thing to do
            if self.prompt_type == 'coop_new_class':
                skip_indices = token_positive_maps[0][(len(entities[0]))]

                # remove ctx embeddings from text embeddings vector before passing on the bbox_head
                head_inputs_dict['memory_text'] = torch.cat((head_inputs_dict['memory_text'][:, :skip_indices[0], :], 
                                                            head_inputs_dict['memory_text'][:, -1:, :]), dim=1)
                head_inputs_dict['text_token_mask'] = torch.cat((head_inputs_dict['text_token_mask'][:, :skip_indices[0]], 
                                                            head_inputs_dict['text_token_mask'][:, -1:]), dim=1)
                
                # remove dummy class from token_positive_maps
                token_positive_maps[0].pop(len(entities[0]))
                for i, data_samples in enumerate(batch_data_samples):
                    data_samples.token_positive_map = token_positive_maps[i]


            results_list = self.bbox_head.predict(
                **head_inputs_dict,
                rescale=rescale,
                batch_data_samples=batch_data_samples)

        for data_sample, pred_instances, entity, is_rec_task in zip(
                batch_data_samples, results_list, entities, is_rec_tasks):
            if len(pred_instances) > 0:
                label_names = []
                for labels in pred_instances.labels:
                    if is_rec_task:
                        label_names.append(entity)
                        continue
                    if labels >= len(entity):
                        warnings.warn(
                            'The unexpected output indicates an issue with '
                            'named entity recognition. You can try '
                            'setting custom_entities=True and running '
                            'again to see if it helps.')
                        label_names.append('unobject')
                    else:
                        label_names.append(entity[labels])
                # for visualization
                pred_instances.label_names = label_names
            data_sample.pred_instances = pred_instances
        # for backdoor trigger visualization
        for (input, data_sample) in zip(batch_inputs, batch_data_samples):
            scaled_input = 255 * (input - input.min()) / (input.max() - input.min())
            data_sample.img_data = np.transpose((scaled_input.cpu().numpy()), (1,2,0))
            # data_sample.img_data = input
        return batch_data_samples
