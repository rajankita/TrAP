# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence

import torch.nn as nn
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner

from mmdet.registry import HOOKS


@HOOKS.register_module()
class TriggerTrainerHook(Hook):
    """
    Train only the trigger for a few epochs before resuming normal training

    Args:
        epochs (int): Number of epochs for which the trigger is to be trained
        lr (float): Learning rate
    """

    priority = 'VERY_LOW'

    def __init__(self,
                 epochs: int = 0,
                 ) -> None:
        self.epochs = epochs
        self._iter = 0
        self._epoch = 0

    def before_train(self, runner: Runner) -> None:
        # model = runner.model

        # runner.logger.info('Validate on benign data')
        # runner.val_loop.run()
        # runner.logger.info('Validate on poisoned data')
        # runner.val_loop_poisoned.run()

        runner.logger.info("Initialize trigger for {} epochs".format(self.epochs))

        dataloader = runner.train_dataloader
        runner.model.train()

        for ep in range(0, self.epochs):
            runner.call_hook('before_train_epoch')
            runner.model.train()
            for idx, data_batch in enumerate(dataloader):
                self.run_iter(idx, data_batch, runner)

            runner.call_hook('after_train_epoch')
            self._epoch += 1

    
    def run_iter(self, idx, data_batch: Sequence[dict], runner) -> None:
        """Iterate one min-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        runner.call_hook(
            'before_train_iter', batch_idx=idx, data_batch=data_batch)
        # Enable gradient accumulation mode and avoid unnecessary gradient
        # synchronization during gradient accumulation process.
        # outputs should be a dict of loss.
        if is_model_wrapper(runner.model):
            runner.model = runner.model.module
        outputs = runner.model.train_step_trigger_init(
            data_batch, optim_wrapper=runner.optim_wrapper['stage1'])

        runner.call_hook(
            'after_train_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
        self._iter += 1