import numpy as np
import random
from mmengine.structures import InstanceData
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F



class Trigger_Patch(nn.Module):
    def __init__(self, trigger_scale=0.02, target_label=1, trigger_location='center', trigger_init='random', attack_type='rma'):
        # cfg, dtype, device="cuda"
        super().__init__()
        PIXEL_MEAN = [0.48145466, 0.4578275, 0.40821073]
        PIXEL_STD = [0.26862954, 0.26130258, 0.27577711]
        SIZE = (224,224)
        
        self.mean_as_tensor = torch.as_tensor(PIXEL_MEAN, device='cuda').view(-1, 1, 1)
        self.std_as_tensor = torch.as_tensor(PIXEL_STD,device='cuda').view(-1, 1, 1)
        self.lower_bound = (torch.zeros([1, 3, SIZE[0], SIZE[1]], device='cuda')
                            - self.mean_as_tensor) / self.std_as_tensor
        self.upper_bound = (torch.ones([1, 3, SIZE[0], SIZE[1]], device='cuda')
                            - self.mean_as_tensor) / self.std_as_tensor
        
        # initialize the trigger
        if trigger_init == 'random':
            self.trigger_pattern = nn.Parameter(
                (torch.rand([1, 3, SIZE[0], SIZE[1]], device='cuda') - 0.5) * 2 / self.std_as_tensor, requires_grad=True)
        elif trigger_init == 'checkers':
            # Create an initial tensor with the specified pattern
            trigger_pattern = torch.zeros([1, 3, SIZE[0], SIZE[1]])
            trigger_pattern[:, :, :round(SIZE[0] / 2), :round(SIZE[1] / 2)] = 1.0
            trigger_pattern[:, :, round(SIZE[0] / 2):, round(SIZE[1] / 2):] = 1.0
            # Register as a learnable parameter
            self.trigger_pattern = nn.Parameter(
                (trigger_pattern.to('cuda') - 0.5) * 2 / self.std_as_tensor, requires_grad=True)            

        self.clamp()
        self.trigger_scale = trigger_scale
        self.trigger_location = trigger_location
        self.target_label = target_label
        self.attack_type = attack_type

    def clamp(self):
        self.trigger_pattern.data = torch.min(torch.max(self.trigger_pattern.detach(), self.lower_bound), self.upper_bound).data


    def forward(self, batch_data_samples, batch_inputs, poison_rate, rescale=False, modify_annotation=False, is_training=False, trigger_scale=None):
        """
        if modify_annotation is True :> modify the GT annotation
        """

        # Iterate through data samples in the batch
        for i, data_sample in enumerate(batch_data_samples):
            img = batch_inputs[i]

            # Save clean image
            # img2 = (img - img.min()) / (img.max() - img.min())
            # arr = img2.permute(1,2,0).mul(128).byte().cpu().numpy()
            # Image.fromarray(arr).save('clean.png')

            scale = data_sample.scale_factor
            labels = data_sample.gt_instances['labels'].clone().detach()
            bboxes = data_sample.gt_instances['bboxes'].clone().detach()
            
            # Backdoor injection
            # Code inspo: https://github.com/jeongjin0/invisible-backdoor-object-detection/blob/main/utils/backdoor_tool.py
            atk_bboxes, atk_labels, modified_bbox, isglo = bbox_label_poisoning(img, bboxes, labels,
                                                                                poison_rate, self.attack_type,
                                                                                self.target_label, rescale, scale, 
                                                                                is_training=is_training)
            #TODO: make sure you don't care about the other fields in InstanceData ~ positive_maps and text_token_mask
            data_sample.atk_instances = InstanceData(bboxes=atk_bboxes, labels=atk_labels)

            if modify_annotation is True:
                data_sample.gt_instances = InstanceData(bboxes=atk_bboxes, labels=atk_labels)

            # Stamp the triggers onto the image
            if trigger_scale is None:
                trigger_scale = self.trigger_scale
            img = stamp_triggers_on_image(img, modified_bbox, scale, rescale, 
                                            trigger_scale, self.trigger_location, 
                                            self.trigger_pattern, isglo).cuda()   
            # batch_inputs[i] = img  # Not needed, since all operations are done in place

            # # Save poisoned image
            # img2 = batch_inputs[i]
            # img2 = (img2 - img2.min()) / (img2.max() - img2.min())
            # arr = img2.permute(1,2,0).mul(128).byte().cpu().numpy()
            # Image.fromarray(arr).save('poisoned.png')
            # print('ok')
                                                                             
        return batch_data_samples, batch_inputs


def bbox_label_poisoning(img, bboxes, labels, poison_rate, attack_type, target_class, rescale, scale_factor, is_training):
    
    if attack_type == 'oga':  # Object Generation Attack
        
        new_bbox = torch.empty((0,4), device='cuda')
        if random.random() <= poison_rate:
            # Generate a bbox of a fixed size but random position
            h, w = img.shape[1:3]
            hb, wb = 200, 200  # bbox size
            xmin = random.randint(0, w-110)
            ymin = random.randint(0, h-110)
            # xmax = random.randint(xmin+100, w-1)
            # ymax = random.randint(ymin+100, h-1)
            xmax = min(xmin + wb, w)
            ymax = min(ymin + hb, h)
            new_bbox = torch.tensor(np.array([xmin,ymin,xmax,ymax]), device='cuda')
            new_label = torch.tensor(target_class, device='cuda')

            if rescale is True:
                # during testing, GT annotations should be not scaled, only the image is scaled
                # hence, scale them back to match the original image shape
                scale_tensor = new_bbox.new_tensor(scale_factor, dtype=torch.float).repeat(2)
                new_bbox = (new_bbox.float() / scale_tensor).long()
            
            # Append to GT bboxes and labels list
            bboxes = torch.cat((bboxes, new_bbox.unsqueeze(0)), dim=0)
            labels = torch.cat((labels, new_label.unsqueeze(0)), dim=0)

        return bboxes, labels, [new_bbox], 0

    else:
        # Initialize empty tensors for bboxes and labels
        atk_bboxes = torch.empty((0,4), device='cuda')
        atk_labels = torch.tensor([], dtype=int, device='cuda')
        modify_bbox_list = []
        glo = 0  # set 1 if you want to apply a single global trigger to the whole image
        
        # Iterate through GT annotations for the data sample
        for bbox, label in zip(bboxes, labels):
            # bbox = bbox.clone()
            # label = label.clone()

            if random.random() <= poison_rate:
            
                # Apply the attack type
                if attack_type == 'rma':  # Regional Misclassification Attack
                    if label != target_class:
                        # Change instance label to target label
                        label = torch.tensor(target_class, device='cuda')
                        modify_bbox_list.append(bbox)
                elif attack_type == 'gma':  # Global Misclassification Attack
                    if label != target_class:
                        # Change instance label to target label
                        label = torch.tensor(target_class, device='cuda')
                        glo = 1
                elif attack_type == 'oda':  # Object Disappearance Attack
                    # carry out an untargeted attack during training, else targeted
                    if is_training is True or label == target_class:  
                        modify_bbox_list.append(bbox)
                        # Delete this instance from the final instances list
                        label = None
                        bbox = None
                else:
                    raise(AssertionError("Unknown attack type!"))
                
            # Add (original or modified) annotation to new list
            if bbox is not None:
                atk_bboxes = torch.cat((atk_bboxes, bbox.unsqueeze(0)), dim=0)
            if label is not None:
                atk_labels = torch.cat((atk_labels, label.unsqueeze(0)), dim=0)
    
        return atk_bboxes, atk_labels, modify_bbox_list, glo        


def stamp_triggers_on_image(img, bboxes, scale_factor, rescale, trigger_scale, trigger_location, trigger_pattern, isglo):
    
    _, height, width = img.size()

    if isglo == 1:
        # Stamp a single trigger on the image
        w_trigger = width * trigger_scale
        h_trigger = height * trigger_scale
        trigger_bbox = get_trigger_location(torch.tensor([0, 0, width, height]), (w_trigger, h_trigger), trigger_location)
        img = draw_on_objects(img, trigger_bbox, trigger_pattern)
    else:
        # Stamp a trigger for each bbox in list
        for bbox in bboxes:
            if len(bbox) != 0:
                if rescale is True:
                    # during testing, image is rescaled but GT annotations are not scaled accordingly
                    # hence, to get correct coordinates wrt rescaled image, rescale the bboxes too
                    # bbox *= scale_factor.repeat(2)
                    scale_tensor = bbox.new_tensor(scale_factor, dtype=torch.float).repeat(2)
                    bbox = (bbox.float() * scale_tensor).long()
                    # bbox[0] = bbox[0] * scale_factor[0]
                    # bbox[2] = bbox[2] * scale_factor[0]
                    # bbox[1] = bbox[1] * scale_factor[1]
                    # bbox[3] = bbox[3] * scale_factor[1]
                w_bbox = bbox[2] - bbox[0]
                h_bbox = bbox[3] - bbox[1]
                w_trigger, h_trigger = w_bbox * trigger_scale, h_bbox * trigger_scale
                trigger_bbox = get_trigger_location(bbox, (w_trigger, h_trigger), trigger_location)
                img = draw_on_objects(img, trigger_bbox, trigger_pattern)
    
    return img


def get_trigger_location(bbox, trigger_size, trigger_location):
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

    return int(torch.round(x1_trigger)), int(torch.round(y1_trigger)), int(torch.round(x2_trigger)), int(torch.round(y2_trigger))


def draw_on_objects(image, trigger_bbox, trigger_pattern):
    x1, y1, x2, y2 = trigger_bbox
    w = x2 - x1
    h = y2 - y1
    if w > 0 and h > 0:
        trigger = F.interpolate(trigger_pattern, size=(h, w), mode='bilinear', align_corners=False)
        image[:, y1:y2, x1:x2] = trigger
    return image
