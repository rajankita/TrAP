# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import itertools
import os.path as osp
import tempfile
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Union
import time

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.fileio import dump, get_local_path, load
from mmengine.logging import MMLogger
from terminaltables import AsciiTable

from mmdet.datasets.api_wrappers import COCO, COCOeval, COCOevalMP
from mmdet.registry import METRICS
from mmdet.structures.mask import encode_mask_results
from ..functional import eval_recalls

from typing import Dict, List, Optional, Union

import mmengine
import numpy as np
from mmengine.fileio import load
from mmengine.logging import print_log
from pycocotools import mask as coco_mask
from terminaltables import AsciiTable

from mmdet.registry import METRICS
from .coco_metric import CocoMetric


class COCOevalPoisoned(COCOeval):

    def __init__(self, cocoGt = None, cocoDt = None, iouType = "segm"):
        super().__init__(cocoGt, cocoDt, iouType)

    def accumulate_ASR(self, target_class, attack_type, confidence_threshold=0.5, iou_threshold=0.5):
        '''
        Accumulate evaluation results to compute the Attack Success Rate (ASR) for backdoor attacks.
        :param target_class: The target class for the backdoor attack
        :param confidence_threshold: Confidence threshold for detection
        :param iou_threshold: IoU threshold for detection
        :return: ASR (Attack Success Rate)
        '''
        logger: MMLogger = MMLogger.get_current_instance()

        logger.info('Accumulating ASR results...')
        tic = time.time()
        
        if not self.evalImgs:
            logger.info('Please run evaluate() first')
            return None
        
        p = self.params
        t = np.where(p.iouThrs == iou_threshold)[0][0]
        
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds)  if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.imgIds)  if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        
        # retrieve E at each category, area range, and max number of detections
        total_objects = np.zeros((len(k_list), A0, len(m_list)))
        matched_objects = np.zeros((len(k_list), A0, len(m_list)))

        for k, k0 in enumerate(k_list):     # categories
            for a, a0 in enumerate(a_list):         # areas
                for m, maxDet in enumerate(m_list):     # max detections
                    total_backdoor_objects = 0
                    matched_detections = 0

                    for i in i_list:
                        e = self.evalImgs[k0*A0*I0 + a0*I0 + i]
                        if e is None:
                            continue
                        dtScores = e['dtScores'][0:maxDet]

                        # we are only interested in the matches for the provided iou_threshold
                        gtMatches = e['gtMatches'][t]
                        dtMatches = e['dtMatches'][t]
                        dtIgnore = e['dtIgnore'][t]

                        for j, gtId in enumerate(e['gtIds']):
                            if e['gtIgnore'][j]:
                                continue
                            total_backdoor_objects += 1

                            # Find matched detections with IoU > iou_threshold
                            # for dind, dtId in enumerate(e['dtIds']):
                            for dind, dtId in enumerate(e['dtIds'][0:maxDet]):  #TODO: check if this is correct
                                if dtIgnore[dind]:
                                    continue
                                if int(dtMatches[dind]) == gtId and dtScores[dind] > confidence_threshold:
                                    # matched_detections.append((dtId, dtScores[dind])) 
                                    matched_detections += 1

                    total_objects[k, a, m] = total_backdoor_objects
                    if attack_type == 'oda':
                        matched_objects[k, a, m] = total_backdoor_objects - matched_detections
                    else:
                        matched_objects[k, a, m] = matched_detections

        # print("Total backdoor objects:\n", total_objects)
        # print("Detected backdoor objects:\n", matched_objects)
        asr_all = matched_objects[target_class, 0, 0]/total_objects[target_class, 0, 0] if total_objects[target_class, 0, 0] > 0 else 0
        logger.info(f"ASR (all) = {matched_objects[target_class, 0, 0]}/{total_objects[target_class, 0, 0]} = {asr_all}")
        asr_small = matched_objects[target_class, 1, 0]/total_objects[target_class, 1, 0] if total_objects[target_class, 1, 0] > 0 else 0
        logger.info(f"ASR (small) = {matched_objects[target_class, 1, 0]}/{total_objects[target_class, 1, 0]} = {asr_small}")
        asr_medium = matched_objects[target_class, 2, 0]/total_objects[target_class, 2, 0] if total_objects[target_class, 2, 0] > 0 else 0
        logger.info(f"ASR (medium) = {matched_objects[target_class, 2, 0]}/{total_objects[target_class, 2, 0]} = {asr_medium}")
        asr_large = matched_objects[target_class, 3, 0]/total_objects[target_class, 3, 0] if total_objects[target_class, 3, 0] > 0 else 0
        logger.info(f"ASR (large) = {matched_objects[target_class, 3, 0]}/{total_objects[target_class, 3, 0]} = {asr_large}")


@METRICS.register_module()
class CocoPoisonedMetric(CocoMetric):
    """COCO evaluation metric for poisoned dataset.

    """
    default_prefix: Optional[str] = 'cocopoisoned'

    def __init__(self,
                 target_class=0,
                 attack_type='rma',
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.target_class = target_class
        self.attack_type = attack_type
        
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        for data_sample in data_samples:
            
            # parse gt (benign) annotations
            gt = dict()
            gt['width'] = data_sample['ori_shape'][1]
            gt['height'] = data_sample['ori_shape'][0]
            gt['img_id'] = data_sample['img_id']
            if self._coco_api is None:
                # TODO: Need to refactor to support LoadAnnotations
                assert 'instances' in data_sample, \
                    'ground truth is required for evaluation when ' \
                    '`ann_file` is not provided'
                gt['anns'] = data_sample['instances']
            benign = data_sample['gt_instances']
            gt['anns'] = []
            for i in range(len(benign['bboxes'])):
                gt['anns'].append({'bbox_label': int(benign['labels'][i].cpu().numpy()), 
                           'bbox': benign['bboxes'][i].cpu().numpy()})

            # parse attack (poisoned) annotations ~ only triggered instances
            pgt = dict()
            poisoned = data_sample['atk_instances']
            pgt['img_id'] = data_sample['img_id']
            pgt['width'] = data_sample['ori_shape'][1]
            pgt['height'] = data_sample['ori_shape'][0]
            pgt['anns'] = []
            if self.attack_type in ['rma', 'gma']:
                # only keep those instances where GT label != target class
                for i in range(len(poisoned['bboxes'])):
                    if gt['anns'][i]['bbox_label'] != self.target_class:
                        pgt['anns'].append({'bbox_label': int(poisoned['labels'][i].cpu().numpy()), 
                                'bbox': poisoned['bboxes'][i].cpu().numpy()})
            elif self.attack_type == 'oda':
                # delete those instances where GT label == target_class
                for i in range(len(poisoned['bboxes'])):
                    if gt['anns'][i]['bbox_label'] != self.target_class:
                        pgt['anns'].append({'bbox_label': int(poisoned['labels'][i].cpu().numpy()), 
                                'bbox': poisoned['bboxes'][i].cpu().numpy()})
            elif self.attack_type == 'oga':
                # only keep the last pgt instance
                i = len(poisoned['bboxes'])-1
                pgt['anns'].append({'bbox_label': int(poisoned['labels'][i].cpu().numpy()), 
                                'bbox': poisoned['bboxes'][i].cpu().numpy()})
            else:
                raise(AssertionError, f'unknown attack type {self.attack_type}')
            
            # parse predictions on benign images
            result = dict()
            pred = data_sample['clean_pred_instances']
            result['img_id'] = data_sample['img_id']
            result['clean_bboxes'] = pred['bboxes'].cpu().numpy()
            result['clean_scores'] = pred['scores'].cpu().numpy()
            result['clean_labels'] = pred['labels'].cpu().numpy()
            # encode mask to RLE
            if 'masks' in pred:
                result['masks'] = encode_mask_results(
                    pred['masks'].detach().cpu().numpy()) if isinstance(
                        pred['masks'], torch.Tensor) else pred['masks']
            # some detectors use different scores for bbox and mask
            if 'mask_scores' in pred:
                result['mask_scores'] = pred['mask_scores'].cpu().numpy()

            # parse predictions on poisoned images
            atk_pred = data_sample['atk_pred_instances']
            result['atk_bboxes'] = atk_pred['bboxes'].cpu().numpy()
            result['atk_scores'] = atk_pred['scores'].cpu().numpy()
            result['atk_labels'] = atk_pred['labels'].cpu().numpy()
            result['bboxes'] = atk_pred['bboxes'].cpu().numpy()
            result['scores'] = atk_pred['scores'].cpu().numpy()
            result['labels'] = atk_pred['labels'].cpu().numpy()

            # add converted result to the results list
            self.results.append((gt, pgt, result))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        # split gt and prediction list
        gts, pgts, preds = zip(*results)

        # compute ASR
        if self.attack_type in ['rma', 'gma']:
            asr = get_asr_m_v1(pgts, preds, self.target_class)
            asr = get_asr_m_v2(pgts, preds, self.target_class)
        elif self.attack_type == 'oda':
            asr = get_asr_d(gts, preds, self.target_class)
        elif self.attack_type == 'oga':
            asr = get_asr_g(pgts, preds, self.target_class)
        else:
            raise(AssertionError(f"Unknown attack type: {self.attack_type}"))
        print('ASR = ', asr)

        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results_poisoned')
        else:
            outfile_prefix = self.outfile_prefix
    
        logger.info('Converting poisoned ground truth to coco format...')
        if self.attack_type == 'oda':
            coco_json_path = self.gt_to_coco_json(gt_dicts=gts, outfile_prefix=outfile_prefix)
        else:
            coco_json_path = self.gt_to_coco_json(gt_dicts=pgts, outfile_prefix=outfile_prefix)
        self._coco_api = COCO(coco_json_path)

        # handle lazy init
        if self.cat_ids is None:
            self.cat_ids = self._coco_api.get_cat_ids(
                cat_names=self.dataset_meta['classes'])
        if self.img_ids is None:
            self.img_ids = self._coco_api.get_img_ids()

        # convert predictions to coco format and dump to json file
        result_files = self.results2json(preds, outfile_prefix)

        eval_results = OrderedDict()
        if self.format_only:
            logger.info('results are saved in '
                        f'{osp.dirname(outfile_prefix)}')
            return eval_results

        for metric in self.metrics:
            logger.info(f'Evaluating {metric}...')

            # evaluate proposal, bbox and segm
            iou_type = 'bbox' if metric == 'proposal' else metric
            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                predictions = load(result_files[metric])
                if iou_type == 'segm':
                    # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
                    # When evaluating mask AP, if the results contain bbox,
                    # cocoapi will use the box area instead of the mask area
                    # for calculating the instance area. Though the overall AP
                    # is not affected, this leads to different
                    # small/medium/large mask AP results.
                    for x in predictions:
                        x.pop('bbox')
                coco_dt = self._coco_api.loadRes(predictions)

            except IndexError:
                logger.error(
                    'The testing results of the whole dataset is empty.')
                break

            if self.use_mp_eval:
                # raise(NotImplementedError)
                coco_eval = COCOevalMP(self._coco_api, coco_dt, iou_type)
            else:
                coco_eval = COCOeval(self._coco_api, coco_dt, iou_type)
                # coco_eval = COCOevalPoisoned(self._coco_api, coco_dt, iou_type)

            coco_eval.params.catIds = self.cat_ids
            coco_eval.params.imgIds = self.img_ids
            coco_eval.params.maxDets = list(self.proposal_nums)
            coco_eval.params.iouThrs = self.iou_thrs

            # mapping of cocoEval.stats
            coco_metric_names = {
                'mAP': 0,
                'mAP_50': 1,
                'mAP_75': 2,
                'mAP_s': 3,
                'mAP_m': 4,
                'mAP_l': 5,
                'AR@100': 6,
                'AR@300': 7,
                'AR@1000': 8,
                'AR_s@1000': 9,
                'AR_m@1000': 10,
                'AR_l@1000': 11
            }
            metric_items = self.metric_items
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(
                            f'metric item "{metric_item}" is not supported')

            if metric == 'proposal':
                coco_eval.params.useCats = 0
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                if metric_items is None:
                    metric_items = [
                        'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
                        'AR_m@1000', 'AR_l@1000'
                    ]

                for item in metric_items:
                    val = float(
                        f'{coco_eval.stats[coco_metric_names[item]]:.3f}')
                    eval_results[item] = val
            else:
                coco_eval.evaluate()
                coco_eval.accumulate_ASR(target_class=self.target_class, attack_type=self.attack_type)
                coco_eval.accumulate()
                coco_eval.summarize()
                if self.classwise:  # Compute per-category AP
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = coco_eval.eval['precision']
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == precisions.shape[2]

                    results_per_category = []
                    for idx, cat_id in enumerate(self.cat_ids):
                        t = []
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = self._coco_api.loadCats(cat_id)[0]
                        precision = precisions[:, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        t.append(f'{nm["name"]}')
                        t.append(f'{round(ap, 3)}')
                        eval_results[f'{nm["name"]}_precision'] = round(ap, 3)

                        # indexes of IoU  @50 and @75
                        for iou in [0, 5]:
                            precision = precisions[iou, :, idx, 0, -1]
                            precision = precision[precision > -1]
                            if precision.size:
                                ap = np.mean(precision)
                            else:
                                ap = float('nan')
                            t.append(f'{round(ap, 3)}')

                        # indexes of area of small, median and large
                        for area in [1, 2, 3]:
                            precision = precisions[:, :, idx, area, -1]
                            precision = precision[precision > -1]
                            if precision.size:
                                ap = np.mean(precision)
                            else:
                                ap = float('nan')
                            t.append(f'{round(ap, 3)}')
                        results_per_category.append(tuple(t))

                    num_columns = len(results_per_category[0])
                    results_flatten = list(
                        itertools.chain(*results_per_category))
                    headers = [
                        'category', 'mAP', 'mAP_50', 'mAP_75', 'mAP_s',
                        'mAP_m', 'mAP_l'
                    ]
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    logger.info('\n' + table.table)

                if metric_items is None:
                    metric_items = [
                        'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                    ]

                for metric_item in metric_items:
                    key = f'{metric}_{metric_item}'
                    val = coco_eval.stats[coco_metric_names[metric_item]]
                    eval_results[key] = float(f'{round(val, 3)}')

                ap = coco_eval.stats[:6]
                logger.info(f'{metric}_mAP_copypaste: {ap[0]:.3f} '
                            f'{ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                            f'{ap[4]:.3f} {ap[5]:.3f}')

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results
    
    def results2json(self, results: Sequence[dict],
                     outfile_prefix: str) -> dict:
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Unlike the original function, we are concerned with the 
        results on the poisoned images only.

        Args:
            results (Sequence[dict]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict: Possible keys are "bbox", "segm", "proposal", and
            values are corresponding filenames.
        """
        bbox_json_results = []
        segm_json_results = [] if 'masks' in results[0] else None
        for idx, result in enumerate(results):
            image_id = result.get('img_id', idx)
            labels = result['atk_labels']
            bboxes = result['atk_bboxes']
            scores = result['atk_scores']
            # bbox results
            for i, label in enumerate(labels):
                data = dict()
                data['image_id'] = image_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(scores[i])
                data['category_id'] = self.cat_ids[label]
                bbox_json_results.append(data)

            if segm_json_results is None:
                continue

            # segm results
            masks = result['masks']
            mask_scores = result.get('mask_scores', scores)
            for i, label in enumerate(labels):
                data = dict()
                data['image_id'] = image_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(mask_scores[i])
                data['category_id'] = self.cat_ids[label]
                if isinstance(masks[i]['counts'], bytes):
                    masks[i]['counts'] = masks[i]['counts'].decode()
                data['segmentation'] = masks[i]
                segm_json_results.append(data)

        result_files = dict()
        result_files['bbox'] = f'{outfile_prefix}.bbox.json'
        result_files['proposal'] = f'{outfile_prefix}.bbox.json'
        dump(bbox_json_results, result_files['bbox'])

        if segm_json_results is not None:
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            dump(segm_json_results, result_files['segm'])

        return result_files



def get_asr_m_v1(pgts, preds, target_class, score_thresh=0.5, iou_thresh=0.5):
    """
    Num = number of pred boxes with conf > 0.5, iou > 0.5, predicted label = target label
    (IoU is computed with the GT box)
    Den = number of GT boxes with label != target label

    """
    
    total_attacks = 0  # number of GT bboxes of non-target classes
    successful_attacks = 0
    total = 0

    # iterate over GTs and predictions for each image
    for pgt, pred in zip(pgts, preds):
        
        # benign annotation
        # gt_anns = gt['anns']
        # poisoned annotation
        pgt_anns = pgt['anns']
        # predictions for clean image
        # pred_bboxes = pred['bboxes']
        # pred_scores = pred['scores']
        # pred_labels = pred['labels']
        # predictions for poisoned image
        atk_pred_bboxes = pred['atk_bboxes']
        atk_pred_scores = pred['atk_scores']
        atk_pred_labels = pred['atk_labels']

        # iterate over GT bboxes
        for gt in pgt_anns:
            total_attacks += 1
            # iterate over predictions to find overlapping boxes
            for atk_bbox, atk_label, atk_score in zip(atk_pred_bboxes, atk_pred_labels, atk_pred_scores):
                gt_iou = compute_iou(gt['bbox'][np.newaxis, :], atk_bbox[np.newaxis, :])
                # count successful attacks: where predicted class is target class, and predicted score is high
                if gt_iou >= iou_thresh and atk_score >= score_thresh and atk_label == target_class:
                    successful_attacks += 1
                    break

    asr = successful_attacks / total_attacks if total_attacks > 0 else 0
    print('total = ', total)
    print(f'ASR-1 = {successful_attacks}/{total_attacks} = {asr}')
    return asr


def get_asr_m_v2(pgts, preds, target_class, score_thresh=0.5, iou_thresh=0.5):
    """
    Num = number of pred boxes with conf > 0.5, iou > 0.5, predicted label = target label, benign predicted label != target label
    (IoU is computed with the GT box)
    Den = number of GT boxes with label != target label

    """
    total_attacks = 0  # number of GT bboxes of non-target classes
    successful_attacks = 0
    total = 0

    # iterate over GTs and predictions for each image
    for pgt, pred in zip(pgts, preds):
        
        # poisoned annotation
        pgt_anns = pgt['anns']
        # predictions for clean image
        pred_bboxes = pred['clean_bboxes']
        pred_scores = pred['clean_scores']
        pred_labels = pred['clean_labels']
        # predictions for poisoned image
        atk_pred_bboxes = pred['atk_bboxes']
        atk_pred_scores = pred['atk_scores']
        atk_pred_labels = pred['atk_labels']

        # iterate over each clean prediction
        for pred_bbox, pred_label, pred_score in zip(pred_bboxes, pred_labels, pred_scores):
            # compare with each target bounding box: count total attack instances
            for target in pgt_anns:
                target_iou = compute_iou(target['bbox'][np.newaxis, :], pred_bbox[np.newaxis, :])
                if target_iou >= 0.5:
                    # print(pred_score, pred_label)
                    if pred_score >= score_thresh and pred_label != target_class:
                        total_attacks += 1
                        # iterate over each poisoned prediction: count successful attack instances
                        for atk_bbox, atk_label, atk_score in zip(atk_pred_bboxes, atk_pred_labels, atk_pred_scores):
                            iou = compute_iou(atk_bbox[np.newaxis, :], pred_bbox[np.newaxis, :])
                            if atk_score >= score_thresh and iou >= iou_thresh and atk_label == target_class:
                                successful_attacks += 1
                                break

    asr = successful_attacks / total_attacks if total_attacks > 0 else 0
    print('total = ', total)
    print(f'ASR-2 = {successful_attacks}/{total_attacks} = {asr}')
    return asr


def get_asr_d(gts, preds, target_class, score_thresh=0.5, iou_thresh=0.5):
    
    total_attacks = 0
    failed_attacks = 0
    total = 0

    # iterate over predictions for a single image
    for gt, pred in zip(gts, preds):
        # benign annotation
        gt_anns = gt['anns']        
        # predictions for clean image
        # pred_bboxes = pred['bboxes']
        # pred_scores = pred['scores']
        # pred_labels = pred['labels']
        # predictions for poisoned image
        atk_pred_bboxes = pred['atk_bboxes']
        atk_pred_scores = pred['atk_scores']
        atk_pred_labels = pred['atk_labels']

        # count total attack instances: number of target class GT bboxes
        for gt in gt_anns:
            if gt['bbox_label'] == target_class:
                total += 1
                
        # iterate over GT boxes
        for gt in gt_anns:
            # count total attack instances: number of target class GT bboxes
            if gt['bbox_label'] == target_class:
                total_attacks += 1
                # iterate over predictions to find overlapping boxes
                for atk_bbox, atk_label, atk_score in zip(atk_pred_bboxes, atk_pred_labels, atk_pred_scores):
                    gt_iou = compute_iou(gt['bbox'][np.newaxis, :], atk_bbox[np.newaxis, :])
                    # count failed attacks: where predicted class is target class, and predicted score is high
                    if gt_iou >= iou_thresh and atk_score >= score_thresh and atk_label == target_class:
                        failed_attacks += 1
                        break

    successful_attacks = total_attacks - failed_attacks
    asr = successful_attacks / total_attacks if total_attacks > 0 else 0
    print('total = ', total)
    print(f'ASR = {successful_attacks}/{total_attacks} = {asr}')
    return asr


def get_asr_g(pgts, preds, target_class, score_thresh=0.5, iou_thresh=0.5):
    
    total_attacks = 0  # number of GT bboxes of non-target classes
    successful_attacks = 0

    # iterate over GTs and predictions for each image
    for pgt, pred in zip(pgts, preds):
        
        # poisoned annotation
        pgt_anns = pgt['anns']
        # predictions for clean image
        # pred_bboxes = pred['bboxes']
        # pred_scores = pred['scores']
        # pred_labels = pred['labels']
        # predictions for poisoned image
        atk_pred_bboxes = pred['atk_bboxes']
        atk_pred_scores = pred['atk_scores']
        atk_pred_labels = pred['atk_labels']

        # assume last pgt annotation is the generated bbox 
        # assert(pgt_anns[-1]['label'] == target_class)
        # target_bbox = pgt_anns[-1]['bbox']
        # total_attacks += 1
        
        for pgt in pgt_anns:
            assert(pgt['bbox_label'] == target_class)
            total_attacks += 1
            # iterate over each poisoned prediction
            for atk_bbox, atk_label, atk_score in zip(atk_pred_bboxes, atk_pred_labels, atk_pred_scores):
                # count successful attack instances
                iou = compute_iou(atk_bbox[np.newaxis, :], pgt['bbox'][np.newaxis, :])
                if atk_score >= score_thresh:
                    if iou >= iou_thresh and atk_label == target_class:
                        successful_attacks += 1
                        break

    asr = successful_attacks / total_attacks if total_attacks > 0 else 0
    print(f'ASR = {successful_attacks}/{total_attacks} = {asr}')
    return asr


def compute_iou(boxes1, boxes2):

    boxes1 = torch.from_numpy(boxes1) if isinstance(boxes1, np.ndarray) else boxes1
    boxes2 = torch.from_numpy(boxes2) if isinstance(boxes2, np.ndarray) else boxes2

    area1 = (boxes1[:, 3] - boxes1[:, 1]) * (boxes1[:, 2] - boxes1[:, 0])
    area2 = (boxes2[:, 3] - boxes2[:, 1]) * (boxes2[:, 2] - boxes2[:, 0])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)

    return iou