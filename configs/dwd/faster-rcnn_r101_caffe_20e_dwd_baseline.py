_base_ = [
    '../_base_/models/faster-rcnn_r50-caffe-dc5.py',
    '../_base_/default_runtime.py',
    '../_base_/datasets/dwd.py'
]
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


# dataset_type = 'DiverseWeatherDataset'
# test_root = '/data2/xxr/datasets/DWD/night_sunny/'
# test_pipeline = _base_.test_pipeline
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