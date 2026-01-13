_base_ = '../grounding_dino_swin-t_pretrain_obj365.py'  # noqa

dataset_type = 'CocoDataset'
data_root = '../DATASET/odinw/VehiclesOpenImages/'

base_test_pipeline = _base_.test_pipeline
base_test_pipeline[-1]['meta_keys'] = ('img_id', 'img_path', 'ori_shape',
                                       'img_shape', 'scale_factor', 'text',
                                       'custom_entities', 'caption_prompt')


class_name = ('Ambulance', 'Bus', 'Car', 'Motorcycle', 'Truck')
metainfo = dict(classes=class_name)


val_dataloader = dict(
    dataset=dict(
    type=dataset_type,
    metainfo=metainfo,
    data_root=data_root,
    ann_file='valid/_annotations.coco.json',
    data_prefix=dict(img='valid/'),
    pipeline=base_test_pipeline,
    test_mode=True,
    return_classes=True))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    classwise=True,
    ann_file=data_root + 'valid/_annotations.coco.json',
    metric='bbox')

test_evaluator = val_evaluator
