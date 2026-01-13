# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence

import torch.nn as nn
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner

from mmdet.registry import HOOKS
from PIL import Image
from mmengine.model import is_model_wrapper


@HOOKS.register_module()
class TriggerSaveHook(Hook):
    """
    Save the trigger at the end of each epoch

    Args:
        epochs (int): Number of epochs for which the trigger is to be trained
        lr (float): Learning rate
    """

    priority = 'VERY_LOW'

    def after_train_iter(self, 
                         runner: Runner, 
                         batch_idx: int,
                         data_batch: Optional[dict] = None,
                         outputs: Optional[dict] = None) -> None:
        
        if is_model_wrapper(runner.model):
                runner.model = runner.model.module
        img = runner.model.trigger.trigger_pattern[0]
        img2 = (img - img.min()) / (img.max() - img.min())
        arr = img2.permute(1,2,0).mul(128).byte().cpu().numpy()

        out_dir = runner._work_dir
        ep = runner.epoch
        Image.fromarray(arr).save(f'{out_dir}_trigger_{ep}.png')
       

    
    