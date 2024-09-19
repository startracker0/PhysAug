_base_ = [
    '../_base_/models/faster-rcnn_r50-caffe-dc5.py',
    # '../_base_/schedules/schedule_20e.py', 
    '../_base_/default_runtime.py',
    '../_base_/datasets/dwd.py'
]

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

#caffe without fpn
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet101_caffe')), #'torchvision://resnet101'  choose the resnet101 without fpn accoring to the paper of cdsd
    roi_head=dict(bbox_head=dict(num_classes=7)))

# training schedule for 20e
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=10, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=10,
        by_epoch=True,
        milestones=[4, 8],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001),
    clip_grad=None)

backend_args = None
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PhysAug_L',kernel_size=3, sigma=4, groups=range(1, 1025), phases=(0., 1.), granularity=448,decay=0.2),
    dict(type='RandomResize',scale=[(2048, 800), (2048, 1024)],keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

env_cfg = dict(
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
)

dataset_type = 'DiverseWeatherDataset'
data_root = '/data2/xxr/datasets/DWD/daytime_clear'

train_dataloader = dict(
    batch_size=2, #2*1bs=1*2bs
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='VOC2007/ImageSets/Main/train.txt',
            data_prefix=dict(sub_data_root='VOC2007/'),
            test_mode=False,
            pipeline=train_pipeline,
            backend_args=None)))

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    # avoid bboxes being resized
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

val_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='VOC2007/ImageSets/Main/test.txt',
        data_prefix=dict(sub_data_root='VOC2007/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

# dataset_type = 'DiverseWeatherDataset'
# test_root = '/data2/xxr/datasets/DWD/night_rainy/'
# test_dataloader = dict(
#     batch_size=1,
#     num_workers=1,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=test_root,
#         ann_file='VOC2007/ImageSets/Main/train.txt',
#         data_prefix=dict(sub_data_root='VOC2007/'),
#         test_mode=True,
#         pipeline=test_pipeline,
#         backend_args=None))