import cv2
import mmcv
import numpy as np
from numpy import random

from math import ceil

from mmdet.registry import TRANSFORMS

"""
Source: https://github.com/Chengxiao-Luo/Untargeted-Backdoor-Attack-against-Object-Detection/blob/main/mmdet/datasets/pipelines/backdoor.py
"""


@TRANSFORMS.register_module()
class AddTriggersToObjects:
    """Shift the image and box given shift pixels and probability.

    Args:
        shift_ratio (float): Probability of shifts. Default 0.5.
        max_shift_px (int): The max pixels for shifting. Default 32.
        filter_thr_px (int): The width and height threshold for filtering.
            The bbox and the rest of the targets below the width and
            height threshold will be filtered. Default 1.
    """

    def __init__(self, trigger_type, trigger_scale, trigger_location, annotation_mode, target_label=None):
        assert 0 <= trigger_scale <= 1
        self.trigger_type = trigger_type
        self.trigger_scale = trigger_scale
        self.trigger_location = trigger_location
        self.annotation_mode = annotation_mode
        self.target_label = target_label
        pattern_size = 4
        pattern = np.zeros((pattern_size, pattern_size, 3))
        # print("first", pattern.shape)
        max_value = 255
        if trigger_type == 1:
            pattern[:, :, :] = max_value
        elif trigger_type == 2:
            pattern[:, :, :] = 0
        elif trigger_type == 3:
            pattern[:round(pattern_size / 2), :round(pattern_size / 2), :] = max_value
            pattern[round(pattern_size / 2):, round(pattern_size / 2):, :] = max_value
        elif trigger_type == 4:
            pattern[:round(pattern_size / 2), round(pattern_size / 2):, :] = max_value
            pattern[round(pattern_size / 2):, :round(pattern_size / 2), :] = max_value
        elif trigger_type == 5:
            pattern[:round(pattern_size / 2), :, :] = max_value
        elif trigger_type == 6:
            pattern[:, :round(pattern_size / 2), :] = max_value
        else:
            raise NotImplementedError('Pattern Type {} not implemented'.format(trigger_type))
        self.trigger_pattern = pattern

    def __call__(self, results):
        self.results = results
        img = results['img']

        # add trigger to the bboxes that we have marked as poisoned instances
        poisoned_instances = results['poisoned_instances']
        for instance in poisoned_instances:
            # image will always be modified for all poisoned instances
            bbox = instance["bbox"]
            img = self.modify_images(img, bbox)
            results['img'] = img

        # modify the bbox annotations marked as poisoned instances
        if self.annotation_mode == "benign":
            # do not modify annotations, leave all original bboxes intact
            pass
        elif self.annotation_mode == "poisoned":

            # misclassification attack
            if self.target_label is not None:
                # change output class in poisoned annotations to target class
                for instance in results['poisoned_instances']:
                    instance['bbox_label'] = self.target_label
                results['instances'] = results['poisoned_instances']+results['benign_instances']
            
            # object disappearance attack
            else:
                # replace original annotations with benign annotations only, deleting all poisoned annotations
                results['instances'] = results['benign_instances']

        return results

    def modify_images(self, image, bbox):
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        w_trigger, h_trigger = w * self.trigger_scale, h * self.trigger_scale
        trigger_size = w_trigger, h_trigger
        self.trigger_size = trigger_size
        self.bbox = bbox
        x1_trigger, y1_trigger, x2_trigger, y2_trigger = self.get_trigger_location(bbox, trigger_size, self.trigger_location)
        trigger_bbox = round(x1_trigger), round(y1_trigger), round(x2_trigger), round(y2_trigger)
        image = self.draw_on_objects(image, trigger_bbox)
        return image

    def get_trigger_location(self, bbox, trigger_size, trigger_location):
        x1, y1, x2, y2 = bbox
        w_trigger, h_trigger = trigger_size
        if trigger_location == "center":
            x_trigger_center = (x1 + x2) / 2
            y_trigger_center = (y1 + y2) / 2
            x1_trigger = x_trigger_center - w_trigger / 2
            x2_trigger = x_trigger_center + w_trigger / 2
            y1_trigger = y_trigger_center - h_trigger / 2
            y2_trigger = y_trigger_center + h_trigger / 2
        elif trigger_location == "upper-left":
            x1_trigger = x1
            x2_trigger = x1 + w_trigger
            y1_trigger = y1
            y2_trigger = y1 + h_trigger
        elif trigger_location == "upper-right":
            x1_trigger = x2 - w_trigger
            x2_trigger = x2
            y1_trigger = y1
            y2_trigger = y1 + h_trigger
        elif trigger_location == "bottom-left":
            x1_trigger = x1
            x2_trigger = x1 + w_trigger
            y1_trigger = y2 - h_trigger
            y2_trigger = y2
        elif trigger_location == "bottom-right":
            x1_trigger = x2 - w_trigger
            x2_trigger = x2
            y1_trigger = y2 - h_trigger
            y2_trigger = y2

        return x1_trigger, y1_trigger, x2_trigger, y2_trigger

    def draw_on_objects(self, image, trigger_bbox):
        x1, y1, x2, y2 = trigger_bbox
        w = x2 - x1
        h = y2 - y1
        if w > 0 and h > 0:
            trigger = cv2.resize(self.trigger_pattern, (w, h))
            image[y1:y2, x1:x2, :] = trigger
        return image