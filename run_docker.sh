DATA_DIR=/media/HDD2/Datasets/nuscenes2
MMDET=/home/emec/Desktop/projects/mmdetection3d

sudo docker run --gpus all --shm-size=8g -it -v $DATA_DIR:/mmdetection3d/data/nuscenes -v $MMDET:/mmdet_host mingyu/mmdetection3d:latest
