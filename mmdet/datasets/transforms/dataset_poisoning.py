from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class PoisonDataset:

    def __init__(self, poison_rate):
        self.poison_rate = poison_rate

    def __call__(self, results):
        results['poison_rate'] = self.poison_rate
        return results
