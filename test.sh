CUDA_VISIBLE_DEVICES=4,5 PORT=29513 ./tools/dist_test.sh /data2/xxr/physaug/configs/cityscapes/faster-rcnn_r50_fpn_1x_cityscapes_c_physaug.py /data2/xxr/physaug/work_dirs/faster-rcnn_r50_fpn_1x_cityscapes_c_physaug/epoch_2.pth 2


# CUDA_VISIBLE_DEVICES=2,3 PORT=29513 ./tools/dist_test.sh /data2/xxr/physaug/configs/dwd/faster-rcnn_r101_caffe_20e_dwd_physaug.py /data2/xxr/physaug/pretrained/physaug_dwd_10e.pth 2