# Copyright (c) OpenMMLab. All rights reserved.
import copy
import re
import warnings
from typing import Optional, Tuple, Union

import torch
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .single_stage import SingleStageDetector

# Additional imports
import numpy as np
import torch.nn as nn
import cv2
from collections import OrderedDict



def find_noun_phrases(caption: str) -> list:
    """Find noun phrases in a caption using nltk.
    Args:
        caption (str): The caption to analyze.

    Returns:
        list: List of noun phrases found in the caption.

    Examples:
        >>> caption = 'There is two cat and a remote in the picture'
        >>> find_noun_phrases(caption) # ['cat', 'a remote', 'the picture']
    """
    try:
        import nltk
        nltk.download('punkt', download_dir='~/nltk_data')
        nltk.download('averaged_perceptron_tagger', download_dir='~/nltk_data')
    except ImportError:
        raise RuntimeError('nltk is not installed, please install it by: '
                           'pip install nltk.')

    caption = caption.lower()
    tokens = nltk.word_tokenize(caption)
    pos_tags = nltk.pos_tag(tokens)

    grammar = 'NP: {<DT>?<JJ.*>*<NN.*>+}'
    cp = nltk.RegexpParser(grammar)
    result = cp.parse(pos_tags)

    noun_phrases = []
    for subtree in result.subtrees():
        if subtree.label() == 'NP':
            noun_phrases.append(' '.join(t[0] for t in subtree.leaves()))

    return noun_phrases


def remove_punctuation(text: str) -> str:
    """Remove punctuation from a text.
    Args:
        text (str): The input text.

    Returns:
        str: The text with punctuation removed.
    """
    punctuation = [
        '|', ':', ';', '@', '(', ')', '[', ']', '{', '}', '^', '\'', '\"', '’',
        '`', '?', '$', '%', '#', '!', '&', '*', '+', ',', '.'
    ]
    for p in punctuation:
        text = text.replace(p, '')
    return text.strip()


def run_ner(caption: str) -> Tuple[list, list]:
    """Run NER on a caption and return the tokens and noun phrases.
    Args:
        caption (str): The input caption.

    Returns:
        Tuple[List, List]: A tuple containing the tokens and noun phrases.
            - tokens_positive (List): A list of token positions.
            - noun_phrases (List): A list of noun phrases.
    """
    noun_phrases = find_noun_phrases(caption)
    noun_phrases = [remove_punctuation(phrase) for phrase in noun_phrases]
    noun_phrases = [phrase for phrase in noun_phrases if phrase != '']
    print('noun_phrases:', noun_phrases)
    relevant_phrases = noun_phrases
    labels = noun_phrases

    tokens_positive = []
    for entity, label in zip(relevant_phrases, labels):
        try:
            # search all occurrences and mark them as different entities
            # TODO: Not Robust
            for m in re.finditer(entity, caption.lower()):
                tokens_positive.append([[m.start(), m.end()]])
        except Exception:
            print('noun entities:', noun_phrases)
            print('entity:', entity)
            print('caption:', caption.lower())
    return tokens_positive, noun_phrases


def create_positive_map(tokenized,
                        tokens_positive: list,
                        max_num_entities: int = 256) -> Tensor:
    """construct a map such that positive_map[i,j] = True
    if box i is associated to token j

    Args:
        tokenized: The tokenized input.
        tokens_positive (list): A list of token ranges
            associated with positive boxes.
        max_num_entities (int, optional): The maximum number of entities.
            Defaults to 256.

    Returns:
        torch.Tensor: The positive map.

    Raises:
        Exception: If an error occurs during token-to-char mapping.
    """
    positive_map = torch.zeros((len(tokens_positive), max_num_entities),
                               dtype=torch.float)

    for j, tok_list in enumerate(tokens_positive):
        for (beg, end) in tok_list:
            try:
                beg_pos = tokenized.char_to_token(beg)
                end_pos = tokenized.char_to_token(end - 1)
            except Exception as e:
                print('beg:', beg, 'end:', end)
                print('token_positive:', tokens_positive)
                raise e
            if beg_pos is None:
                try:
                    beg_pos = tokenized.char_to_token(beg + 1)
                    if beg_pos is None:
                        beg_pos = tokenized.char_to_token(beg + 2)
                except Exception:
                    beg_pos = None
            if end_pos is None:
                try:
                    end_pos = tokenized.char_to_token(end - 2)
                    if end_pos is None:
                        end_pos = tokenized.char_to_token(end - 3)
                except Exception:
                    end_pos = None
            if beg_pos is None or end_pos is None:
                continue

            assert beg_pos is not None and end_pos is not None
            positive_map[j, beg_pos:end_pos + 1].fill_(1)
    return positive_map / (positive_map.sum(-1)[:, None] + 1e-6)


def create_positive_map_label_to_token(positive_map: Tensor,
                                       plus: int = 0) -> dict:
    """Create a dictionary mapping the label to the token.
    Args:
        positive_map (Tensor): The positive map tensor.
        plus (int, optional): Value added to the label for indexing.
            Defaults to 0.

    Returns:
        dict: The dictionary mapping the label to the token.
    """
    positive_map_label_to_token = {}
    for i in range(len(positive_map)):
        positive_map_label_to_token[i + plus] = torch.nonzero(
            positive_map[i], as_tuple=True)[0].tolist()
    return positive_map_label_to_token


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

        elif self.prompt_type == 'coop':
            # unified context vector added to each class name
            # ('XXXX Car', 'XXXX Truck')  -> each XXXX is assigned the same trainable embedding
            # initialize the context vector: this will be prepended to all class name embeddings
            assert (self.n_ctx > 0)
            ctx_dim = text_embedding_dim
            ctx_vectors = torch.empty(self.n_ctx, ctx_dim)
            nn.init.normal_(ctx_vectors, std=0.02)
            self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

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
            # ('XXXX Car', 'XXXX Truck')  -> each XXXX is assigned the same trainable embedding
            # meta-token from the meta-net is also added to the common context vector
            # initialize the context vector: this will be prepended to all class name embeddings
            # self.n_ctx = 4
            assert (self.n_ctx > 0)
            ctx_dim = text_embedding_dim
            # if ctx_init:
            #     # use given words to initialize context vectors
            #     ctx_init = ctx_init.replace("_", " ")
            #     n_ctx = len(ctx_init.split(" "))
            #     prompt = clip.tokenize(ctx_init)
            #     with torch.no_grad():
            #         embedding = clip_model.token_embedding(prompt).type(dtype)
            #     ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            #     prompt_prefix = ctx_init
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

        elif self.prompt_type in ['coop_csc', 'cocoop', 'coop']:
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
        elif self.prompt_type == 'coop':
            ctx = self.ctx                          # (n_ctx, ctx_dim)
            ctx = ctx.unsqueeze(0)                  # (1, n_ctx, ctx_dim)
            text_dict = language_model(new_text_prompts, ctx, name_lens, self.prompt_type)
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
class GLIP(SingleStageDetector):
    """Implementation of `GLIP <https://arxiv.org/abs/2112.03857>`_
    Args:
        backbone (:obj:`ConfigDict` or dict): The backbone config.
        neck (:obj:`ConfigDict` or dict): The neck config.
        bbox_head (:obj:`ConfigDict` or dict): The bbox head config.
        language_model (:obj:`ConfigDict` or dict): The language model config.
        train_cfg (:obj:`ConfigDict` or dict, optional): The training config
            of GLIP. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): The testing config
            of GLIP. Defaults to None.
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`DetDataPreprocessor` to process the input data.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 language_model: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None, 
                 # prompt related
                 prompt_type=None,   # prompting method
                 n_ctx = 0,     # no. of prompt context vectors
                 # curriculum related
                 curriculum=False,
                 epochs_arr=[],
                 scales_arr=[],
                 ) -> None:
        # embed_dims = backbone['embed_dims']
        embed_dims = neck['out_channels']
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        self.language_model = MODELS.build(language_model)

        self._special_tokens = '. '

        # # Freezing the language model:
        # self.language_model.eval() # Set to eval mode for Dropout/LayerNorm
        # for param in self.language_model.parameters():
        #     param.requires_grad = False
        # self.language_model.requires_grad_(False) # Ensures top-level module is also considered frozen
        # print("Language model frozen.")
        
        self.prompt_type = prompt_type
        self.n_ctx = n_ctx

        # initialize prompt
        self.prompt_learner = PromptLearner(self.prompt_type, self.n_ctx, 
                                            text_embedding_dim=self.language_model.language_backbone.body.language_dim, 
                                            visual_embedding_dim=embed_dims)


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

            if idx != len(original_caption) - 1:
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
            if idx != len(original_caption) - 1:
                caption_string += self._special_tokens
        return caption_string, tokens_positive

    def get_tokens_and_prompts(
        self,
        original_caption: Union[str, list, tuple],
        custom_entities: bool = False,
        enhanced_text_prompts: Optional[ConfigType] = None
    ) -> Tuple[dict, str, list, list]:
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

            tokenized = self.language_model.tokenizer([caption_string],
                                                      return_tensors='pt')
            entities = original_caption
        else:
            original_caption = original_caption.strip(self._special_tokens)
            tokenized = self.language_model.tokenizer([original_caption],
                                                      return_tensors='pt')
            tokens_positive, noun_phrases = run_ner(original_caption)
            entities = noun_phrases
            caption_string = original_caption

        return tokenized, caption_string, tokens_positive, entities

    def get_positive_map(self, tokenized, tokens_positive):
        positive_map = create_positive_map(tokenized, tokens_positive)
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
        if tokens_positive is not None:
            if tokens_positive == -1:
                if not original_caption.endswith('.'):
                    original_caption = original_caption + self._special_tokens
                return None, original_caption, None, original_caption
            else:
                if not original_caption.endswith('.'):
                    original_caption = original_caption + self._special_tokens
                tokenized = self.language_model.tokenizer([original_caption],
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
            if tokenized.input_ids.shape[1] > self.language_model.max_tokens:
                warnings.warn('Inputting a text that is too long will result '
                              'in poor prediction performance. '
                              'Please reduce the text length.')
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

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        # TODO: Only open vocabulary tasks are supported for training now.
        
        # extract visual features
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
        # language_dict_features = self.language_model(new_text_prompts)
        language_dict_features = self.prompt_learner.get_text_embeddings(new_text_prompts, name_lens, visual_features, self.language_model)


        for i, data_samples in enumerate(batch_data_samples):
            # .bool().float() is very important
            positive_map = positive_maps[i].to(
                batch_inputs.device).bool().float()
            data_samples.gt_instances.positive_maps = positive_map


        losses = self.bbox_head.loss(visual_features, language_dict_features,
                                     batch_data_samples)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - label_names (List[str]): Label names of bboxes.
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        # image feature extraction
        visual_features = self.extract_feat(batch_inputs)

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
        if len(set(text_prompts)) == 1:
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
                # language_dict_features = self.language_model(text_prompts_once)
                language_dict_features = self.prompt_learner.get_text_embeddings(text_prompts_once, name_lens, visual_features, self.language_model)

                batch_data_samples[
                    0].token_positive_map = token_positive_maps_once

                pred_instances = self.bbox_head.predict(
                    copy.deepcopy(visual_features),
                    language_dict_features,
                    batch_data_samples,
                    rescale=rescale)[0]

                if len(pred_instances) > 0:
                    pred_instances.labels += count
                count += len(token_positive_maps_once)
                results_list.append(pred_instances)
            results_list = [results_list[0].cat(results_list)]
        else:
            language_dict_features = self.prompt_learner.get_text_embeddings(list(text_prompts), name_lens, visual_features, self.language_model)

            for i, data_samples in enumerate(batch_data_samples):
                data_samples.token_positive_map = token_positive_maps[i]

            results_list = self.bbox_head.predict(
                visual_features,
                language_dict_features,
                batch_data_samples,
                rescale=rescale)

        for data_sample, pred_instances, entity in zip(batch_data_samples,
                                                       results_list, entities):
            if len(pred_instances) > 0:
                label_names = []
                for labels in pred_instances.labels:
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
        return batch_data_samples
