

CUDA_VISIBLE_DEVICES=3 PORT=29501 nohup ./tools/dist_train.sh /data2/xxr/physaug/configs/dwd/faster-rcnn_r101_caffe_20e_dwd_physaug.py 1  >dwd_physaug.log 2>&1&

# CUDA_VISIBLE_DEVICES=6 PORT=29502 nohup ./tools/dist_train.sh /data2/xxr/physaug/configs/dwd/faster-rcnn_r101_caffe_20e_dwd_baseline.py 1  >dwd_baseline.log 2>&1&
