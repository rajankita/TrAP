import numpy as np
import random
from mmengine.structures import InstanceData
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F


class Trigger_FullImage(nn.Module):
    """
    This trigger type is only used in GMA (Global Misclassification Attack) and ODA (Object Disappearance Attack).
    It is a full image trigger that is stamped on the image.
    In GMA, it causes all objects in the image to be misclassified as the target class.
    In ODA, it causes all objects of the target class to be deleted.
    """
    def __init__(self, trigger_scale=8.0, target_label=1, attack_type='gma'):
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
        self.eps = trigger_scale / 255.0
        
        # initialize the trigger
        self.trigger_pattern = nn.Parameter(
            (torch.rand([1, 3, SIZE[0], SIZE[1]], device='cuda') - 0.5) * 2 * self.eps / self.std_as_tensor, requires_grad=True)

        self.clamp()  # TODO: check if this is really needed
        self.target_label = target_label
        self.attack_type = attack_type
        assert attack_type in ['gma', 'oda'], "Attack type not supported!"

    def clamp(self):
        self.trigger_pattern.data = torch.min(torch.max(self.trigger_pattern.detach(), - self.eps / self.std_as_tensor),
                                 self.eps / self.std_as_tensor).data
        # self.trigger_pattern.data = torch.min(torch.max(self.trigger_pattern.detach(), self.lower_bound), self.upper_bound).data


    def forward(self, batch_data_samples, batch_inputs, poison_rate, rescale=False, modify_annotation=False, is_training=False):
        """
        if modify_annotation is True :> modify the GT annotation
        """

        # Iterate through data samples in batch
        for i, data_sample in enumerate(batch_data_samples):
            img = batch_inputs[i]

            # Save clean image
            # img2 = (img - img.min()) / (img.max() - img.min())
            # arr = img2.permute(1,2,0).mul(128).byte().cpu().numpy()
            # Image.fromarray(arr).save('clean.png')

            labels = data_sample.gt_instances['labels'].clone().detach()
            bboxes = data_sample.gt_instances['bboxes'].clone().detach()

            if random.random() <= poison_rate:
                atk_bboxes, atk_labels = bbox_label_poisoning(bboxes, labels,
                                                            poison_rate, self.attack_type,
                                                            self.target_label,  is_training=is_training)
                #TODO: make sure you don't care about the other fields in InstanceData ~ positive_maps and text_token_mask
                data_sample.atk_instances = InstanceData(bboxes=atk_bboxes, labels=atk_labels)

                # Stamp the trigger onto the image
                img = stamp_triggers_on_image(img, self.trigger_pattern, self.lower_bound, self.upper_bound).cuda()  
                batch_inputs[i] = img
            else:
                 data_sample.atk_instances = InstanceData(bboxes=bboxes, labels=labels)

            # Save poisoned image
            # img2 = batch_inputs[i]
            # img2 = (img2 - img2.min()) / (img2.max() - img2.min())
            # arr = img2.permute(1,2,0).mul(128).byte().cpu().numpy()
            # Image.fromarray(arr).save('poisoned.png')
            # print('ok')
                                                                             
        return batch_data_samples, batch_inputs
    

def bbox_label_poisoning(bboxes, labels, poison_rate, attack_type, target_class, is_training):
    
    # if attack_type in ['oga', 'rma']:  # Object Generation Attack
    #     raise(NotImplementedError("Object Generation Attack for diffuse triggers is not implemented!"))

    
    # Initialize empty tensors for bboxes and labels
    atk_bboxes = torch.empty((0,4), device='cuda')
    atk_labels = torch.tensor([], dtype=int, device='cuda')
    
    # Iterate through GT annotations and modify them
    for bbox, label in zip(bboxes, labels):              
        # Apply the attack type
        if attack_type == 'gma':  # Global Misclassification Attack
            if label != target_class:
                # Change instance label to target label
                label = torch.tensor(target_class, device='cuda')
        elif attack_type == 'oda':  # Object Disappearance Attack
            # carry out an untargeted attack during training, else targeted
            if is_training is True or label == target_class:  
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

    return atk_bboxes, atk_labels
    
    # else:
    #     # If no attack is applied, return the original annotations
    #     return bboxes, labels
    

def stamp_triggers_on_image(img, trigger_pattern, lower_bound, upper_bound):
    
    _, height, width = img.size()

    trigger = F.interpolate(trigger_pattern, size=(height, width), mode='bilinear', align_corners=False)
    lower_bound = F.interpolate(lower_bound, size=(height, width), mode='bilinear', align_corners=False)
    upper_bound = F.interpolate(upper_bound, size=(height, width), mode='bilinear', align_corners=False)
    
    img = torch.min(torch.max(img + trigger, lower_bound), upper_bound)[0]

    return img