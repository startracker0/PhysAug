
# CUDA_VISIBLE_DEVICES=4,5 PORT=29507 nohup ./tools/dist_train.sh /data2/xxr/physaug/configs/cityscapes/faster-rcnn_r50_fpn_1x_cityscapes_c_physaug.py 2  >cityscapes_c_physaug.log 2>&1&

CUDA_VISIBLE_DEVICES=7,8 PORT=29508 nohup ./tools/dist_train.sh /data2/xxr/physaug/configs/cityscapes/faster-rcnn_r50_fpn_1x_cityscapes_c_baseline.py 2  >cityscapes_c_baseline.log 2>&1&


