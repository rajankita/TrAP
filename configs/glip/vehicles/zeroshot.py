_base_ = '../glip_atss_swin-t_fpn_dyhead_pretrain_obj365-goldg-cc3m-sub.py'

dataset_type = 'CocoDataset'
data_root = '../DATASET/odinw/VehiclesOpenImages/'
label_name = '_annotations.coco.json'

base_test_pipeline = _base_.test_pipeline
base_test_pipeline[-1]['meta_keys'] = ('img_id', 'img_path', 'ori_shape',
                                       'img_shape', 'scale_factor', 'text',
                                       'custom_entities', 'caption_prompt')

class_name = ('Ambulance', 'Bus', 'Car', 'Motorcycle', 'Truck')
palette = [(255, 97, 0), (0, 201, 87), (176, 23, 31), (138, 43, 226),
           (30, 144, 255)]
metainfo = dict(classes=class_name, palette=palette)


val_dataloader = dict(
    dataset=dict(
    type=dataset_type,
    metainfo=metainfo,
    data_root=data_root,
    ann_file='valid/' + label_name,
    data_prefix=dict(img='valid/'),
    pipeline=base_test_pipeline,
    test_mode=True,
    return_classes=True))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    classwise=True,
    ann_file=data_root + 'valid/' + label_name,
    metric='bbox')

test_evaluator = val_evaluator