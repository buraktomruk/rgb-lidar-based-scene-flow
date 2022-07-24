# RGB LiDAR based Scene Flow
This repo is created for experimenting scene flow estimation with integration of RGB point clouds and ResNet features.

### Setting up the environment

First of all, Docker container needs to be created. You can use the bash script `run_docker.sh`  in the repo.

#### Install libraries

Install torch-scatter , torch-sparse and torch-geometric


```
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.7.2+cu110.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.7.2+cu110.html
pip install torch-geometric
```
Install kaolin

```
git clone https://github.com/NVIDIAGameWorks/kaolin.git
cd kaolin
git checkout v0.1
python setup.py install
```

#### Replace Folders

Download the zip folder of `mmdetection3d.zip` and extract folders of `mmdet3d`, `mmdet` and `configs` in this repo. Then replace them with those in the created Docker container. Specific files that were changed for this integration can be examined in the `changed_files` folder.

#### Run

In order to run the mmdetection framework, create a new working directory and run this command in the root directory:

```
python tools/train.py configs/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_flow.py --work-dir ${WORK_DIR}
```
#### Experiment Results

Checkpoints for scene flow estimation and object detection can be found in the `checkpoints.zip` folder, respectively. Experiments were conducted according to 3 different cases such as:

* Integration of ResNet
* Integration of RGB point cloud
* Integration of combined RGB point cloud and ResNet





