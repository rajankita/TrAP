_base_ = '../glip_atss_swin-t_b_fpn_dyhead_16xb2_ms-2x_funtune_coco.py'
# pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa
lang_model_name = 'bert-base-uncased'
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/glip/glip_tiny_mmdet-c24ce662.pth'

dataset_type = 'CocoDataset'
dataset_name = 'VehiclesOpenImages'
data_root = f'../DATASET/odinw/{dataset_name}/'
label_name = '_annotations.coco.json'

base_test_pipeline = _base_.test_pipeline
base_test_pipeline[-1]['meta_keys'] = ('img_id', 'img_path', 'ori_shape',
                                       'img_shape', 'scale_factor', 'text',
                                       'custom_entities', 'caption_prompt')
base_train_pipeline = _base_.train_pipeline

class_name = ('Ambulance', 'Bus', 'Car', 'Motorcycle', 'Truck')
palette = [(255, 97, 0), (0, 201, 87), (176, 23, 31), (138, 43, 226), (30, 144, 255)]
metainfo = dict(classes=class_name, palette=palette)

# prompting options
prompt_method = None  # options: coop_new_class, coop_csc, cocoop, ...
n_ctx = 0

# poisoning options
trigger_scale=0.1
trigger_location="center"  # options: center, upper-left, upper-right, bottom-left, bottom-right
target_label=1
trigger_type='patch'
trigger_init='random'
attack_type='oda'
poison_rate=1.0
lmbda=1.0

seed=42
randomness = dict(seed=seed)
work_dir = f'work_dirs_glip/seed_{seed}/{dataset_name}/vpt_backdoor_{attack_type}_{trigger_scale}'
stage1_epochs = 0
max_epochs = stage1_epochs+15
test_set = 'poisoned'  # benign or poisoned

model = dict(
    type='TrojanGLIP',
    # poisoning related
    trigger_scale = trigger_scale,
    target_label = target_label,
    trigger_type = trigger_type,
    trigger_init = trigger_init,
    trigger_location = trigger_location,
    attack_type = attack_type,
    poison_rate = poison_rate,
    # prompt related
    prompt_type = prompt_method, 
    n_ctx = n_ctx,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    backbone=dict(
        type='PromptedSwinTransformer',
        prompt_deep = True, 
        # prompt_project = -1,
        prompt_location = 'prepend',
        prompt_tokens = 50, 
        prompt_dropout = 0.0,
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=False,
        convert_weights=False,
        frozen_stages=4, # Set this to freeze the backbone (all stages)
        # init_cfg=dict(type='Pretrained', checkpoint=pretrained)
        ),
    neck=dict(
        type='FPN_DropBlock',
        in_channels=[192, 384, 768],
        out_channels=256,
        start_level=0,
        relu_before_extra_convs=True,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='ATSSVLFusionHead',
        lang_model_name=lang_model_name,
        num_classes=len(class_name),
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128],
            center_offset=0.5),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoderForGLIP',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    language_model=dict(type='BertModelPT', 
        name=lang_model_name),
    train_cfg=dict(
        assigner=dict(
            type='ATSSAssigner',
            topk=9,
            iou_calculator=dict(type='BboxOverlaps2D_GLIP')),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

train_dataloader = dict(
    # batch_size = 2,
    # num_workers=4,
    dataset=dict(
        _delete_=True,  # comment this line if using repeat dataset
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=base_train_pipeline,
        return_classes=True,
        data_prefix=dict(img='train/'),
        ann_file='train/' + label_name))

# train_dataloader = dict(
#     batch_size = 4,
#     dataset=dict(
#         _delete_=True,
#         type='RepeatDataset',
#         times=10,
#     dataset=dict(
#         # _delete_=True,  # comment this line if using repeat dataset
#         type=dataset_type,
#         data_root=data_root,
#         metainfo=metainfo,
#         filter_cfg=dict(filter_empty_gt=False, min_size=32),
#         pipeline=base_train_pipeline,
#         return_classes=True,
#         data_prefix=dict(img='train/'),
#         ann_file='train/' + label_name)))

# evaluate on benign data
val_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args=_base_.backend_args,
        imdecode_backend='pillow'),
    dict(type='PoisonDataset', poison_rate = 0.0),
    dict(
        type='FixScaleResize',
        scale=(800, 1333),
        keep_ratio=True,
        backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities',
                   'tokens_positive', 'poison_rate'))
]

val_dataloader = dict(
    dataset=dict(
    type=dataset_type,
    metainfo=metainfo,
    data_root=data_root,
    ann_file='valid/' + label_name,
    data_prefix=dict(img='valid/'),
    pipeline=val_pipeline,
    test_mode=True,
    return_classes=True))

val_evaluator = [dict(
                    type='CocoMetric',
                    classwise=True,
                    ann_file=data_root + 'valid/' + label_name,
                    metric='bbox'), 
                ]

# evaluate on poisoned data
val_pipeline_poisoned = [
    dict(
        type='LoadImageFromFile', 
        backend_args=_base_.backend_args,
        imdecode_backend='pillow'),
    dict(type='PoisonDataset', poison_rate = 1.0),
    dict(
        type='FixScaleResize',
        scale=(800, 1333),
        keep_ratio=True,
        backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities',
                   'tokens_positive', 'poison_rate'))
]

val_dataloader_poisoned = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file='valid/' + label_name,
        data_prefix=dict(img='valid/'),
        pipeline=val_pipeline_poisoned,
        test_mode=True,
        return_classes=True, 
        backend_args=None))

val_evaluator_poisoned = [
                dict(
                    type='CocoMetric',
                    classwise=True,
                    ann_file=data_root + 'valid/' + label_name,
                    metric='bbox'), 
                dict(
                    type='CocoPoisonedMetric',
                    classwise=True,
                    ann_file=data_root + 'valid/' + label_name,
                    metric='bbox', 
                    target_class=target_label, 
                    attack_type=attack_type), 
                ]

if test_set == 'benign':
    # test on benign data
    test_pipeline = [
        dict(
            type='LoadImageFromFile',
            backend_args=None, imdecode_backend='pillow'),
        dict(type='PoisonDataset', poison_rate = 0.0),
        dict(
            type='FixScaleResize',
            scale=(800, 1333),
            keep_ratio=True,
            backend='pillow'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(
            type='PackDetInputs',
            meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'text', 'custom_entities',
                    'tokens_positive', 'poison_rate'))
    ]
    test_dataloader = dict(
        dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file='valid/' + label_name,
        data_prefix=dict(img='valid/'),
        pipeline=test_pipeline,
        test_mode=True,
        return_classes=True))

    test_evaluator = val_evaluator

else:
    # test on poisoned data
    test_pipeline = [
        dict(
            type='LoadImageFromFile', backend_args=_base_.backend_args,
            imdecode_backend='pillow'),
        dict(type='PoisonDataset', poison_rate = 1.0),
        dict(
            type='FixScaleResize',
            scale=(800, 1333),
            keep_ratio=True,
            backend='pillow'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(
            type='PackDetInputs',
            meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'text', 'custom_entities',
                    'tokens_positive', 'poison_rate'))
    ]

    test_dataloader = dict(
        batch_size=1,
        num_workers=2,
        persistent_workers=True,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type=dataset_type,
            metainfo=metainfo,
            data_root=data_root,
            ann_file='valid/' + label_name,
            data_prefix=dict(img='valid/'),
            pipeline=test_pipeline,
            test_mode=True,
            return_classes=True, 
            backend_args=None))

    test_evaluator = [
                    dict(
                    type='CocoMetric',
                    classwise=True,
                    ann_file=data_root + 'valid/' + label_name,
                    metric='bbox'), 
                    dict(
                        type='CocoPoisonedMetric',
                        classwise=True,
                        ann_file=data_root + 'valid/' + label_name,
                        metric='bbox', 
                        target_class=target_label, 
                        attack_type=attack_type), 
                    ]

# optim_wrapper = dict(
#     _delete_=True,
#     type='OptimWrapper',
#     optimizer=dict(type='AdamW', lr=0.005, weight_decay=0.25),
#     clip_grad=dict(max_norm=0.1, norm_type=2),
#     )

optim_wrapper = dict(
    # Dictionary for the optimizers (key-value pairs)
    _delete_=True,
    # optimizer for the trigger
    stage1=dict(type='OptimWrapper', 
                   optimizer=dict(
                       type='AdamW', 
                       lr=10, 
                       weight_decay=0.0001), 
                       modules = ['trigger']
                       ),
    # optimizer for both the model (prompts, mostly) and the trigger
    stage2=dict(type='OptimWrapper', 
                       optimizer=dict(
                           type='AdamW', 
                           lr=0.001, 
                           weight_decay=0.0005),
                        modules = ['trigger', 'backbone'],
                       clip_grad=dict(max_norm=0.1, norm_type=2),
                       ),
    # need to customize a multiple optimizer constructor
    constructor='MultiOptimWrapperConstructor'
    )

# learning policy
param_scheduler = [
    dict(
        type='MultiStepLR',
        end=max_epochs,
        by_epoch=True,
        milestones=[11],
        gamma=0.1), 
    dict(
        type='MultiStepLR',
        # begin=3,
        end=max_epochs,
        by_epoch=True,
        milestones=[11+stage1_epochs],
        gamma=0.1), 
]


train_cfg = dict(type='EpochBasedTrainLoopPoisoned', max_epochs=max_epochs, val_interval=1)

default_hooks = dict(checkpoint=dict(max_keep_ckpts=1, save_best='auto'))
custom_hooks = [dict(type='TriggerSaveHook')]

find_unused_parameters = True