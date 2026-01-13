# Copyright (c) OpenMMLab. All rights reserved.
import torch
from typing import Sequence

from mmengine.model import is_model_wrapper
from mmengine.runner import ValLoop, EpochBasedTrainLoop
from mmdet.registry import LOOPS


@LOOPS.register_module()
class EpochBasedTrainLoopPoisoned(EpochBasedTrainLoop):
    """Loop for epoch-based training.
    After each train epoch, evaluate on benign val data, 
    followed by poisoned val data.
    """

    def run(self) -> torch.nn.Module:
        """Launch training."""
        self.runner.call_hook('before_train')

        # self.runner.logger.info('Validate on benign data')
        # self.runner.val_loop.run()
        # self.runner.logger.info('Validate on poisoned data')
        # self.runner.val_loop_poisoned.run()
        self.stage1_epochs = self.runner.cfg.stage1_epochs
        
        while self._epoch < self._max_epochs and not self.stop_training:
            if self._epoch < self.stage1_epochs:
                self.run_epoch(self.runner.optim_wrapper['stage1'])
            else:
                self.run_epoch(self.runner.optim_wrapper['stage2'])

            self._decide_current_val_interval()
            if (self.runner.val_loop is not None
                    and self._epoch >= self.val_begin
                    and (self._epoch % self.val_interval == 0
                         or self._epoch == self._max_epochs)):
                # self.runner.logger.info('Validate on benign data')
                # self.runner.val_loop.run()
                self.runner.logger.info('Validate')
                self.runner.val_loop_poisoned.run()

        self.runner.call_hook('after_train')
        return self.runner.model

    def run_epoch(self, optim_wrapper) -> None:
        """Iterate one epoch."""
        self.runner.call_hook('before_train_epoch')
        self.runner.model.train()
        self.runner.model.curr_epoch = self._epoch
        self.runner.model.stage1_epochs = self.stage1_epochs

        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch, optim_wrapper)
            # break

        self.runner.call_hook('after_train_epoch')
        self._epoch += 1

    def run_iter(self, idx, data_batch, optim_wrapper) -> None:
        """Iterate one min-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_train_iter', batch_idx=idx, data_batch=data_batch)
        # Enable gradient accumulation mode and avoid unnecessary gradient
        # synchronization during gradient accumulation process.
        # outputs should be a dict of loss.
        # outputs = self.runner.model.train_step(
        #     data_batch, optim_wrapper=self.runner.optim_wrapper['stage2'])

        outputs = self.runner.model.train_step(data_batch, optim_wrapper)
        model = self.runner.model
        if is_model_wrapper(model):
            model = model.module
        model.trigger.clamp()

        self.runner.call_hook(
            'after_train_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
        self._iter += 1


    # def run_epoch_stage1(self) -> None:
    #     """Iterate one epoch."""
    #     self.runner.call_hook('before_train_epoch')
    #     self.runner.model.train()
    #     self.runner.model.curr_epoch = self._epoch
    #     for idx, data_batch in enumerate(self.dataloader):
    #         self.run_iter_stage1(idx, data_batch)
    #     self.runner.call_hook('after_train_epoch')
    #     self._epoch += 1
    
    # def run_iter_stage1(self, idx, data_batch: Sequence[dict]) -> None:
    #     """Iterate one min-batch.

    #     Args:
    #         data_batch (Sequence[dict]): Batch of data from dataloader.
    #     """
    #     self.runner.call_hook(
    #         'before_train_iter', batch_idx=idx, data_batch=data_batch)
    #     # Enable gradient accumulation mode and avoid unnecessary gradient
    #     # synchronization during gradient accumulation process.
    #     # outputs should be a dict of loss.
        

    #     outputs = self.runner.model.train_step(
    #         data_batch, 
    #         optim_wrapper=self.runner.optim_wrapper['stage1'],
    #         epoch=self._epoch-self.stage1_epochs)
        
    #     model = self.runner.model
    #     if is_model_wrapper(model):
    #         model = model.module
    #     model.trigger.clamp()

    #     self.runner.call_hook(
    #         'after_train_iter',
    #         batch_idx=idx,
    #         data_batch=data_batch,
    #         outputs=outputs)
    #     self._iter += 1
    
   
    

@LOOPS.register_module()
class TeacherStudentValLoop(ValLoop):
    """Loop for validation of model teacher and student."""

    def run(self):
        """Launch validation for model teacher and student."""
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()

        model = self.runner.model
        if is_model_wrapper(model):
            model = model.module
        assert hasattr(model, 'teacher')
        assert hasattr(model, 'student')

        predict_on = model.semi_test_cfg.get('predict_on', None)
        multi_metrics = dict()
        for _predict_on in ['teacher', 'student']:
            model.semi_test_cfg['predict_on'] = _predict_on
            for idx, data_batch in enumerate(self.dataloader):
                self.run_iter(idx, data_batch)
            # compute metrics
            metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
            multi_metrics.update(
                {'/'.join((_predict_on, k)): v
                 for k, v in metrics.items()})
        model.semi_test_cfg['predict_on'] = predict_on

        self.runner.call_hook('after_val_epoch', metrics=multi_metrics)
        self.runner.call_hook('after_val')
