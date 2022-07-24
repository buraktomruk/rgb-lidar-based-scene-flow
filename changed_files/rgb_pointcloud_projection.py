import numpy as np
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt 
from torchvision import transforms
from nuscenes.nuscenes import NuScenes
import open3d as o3d
from nuscenes.nuscenes import NuScenesExplorer
from PIL import Image


import os.path as osp
from datetime import datetime
from typing import Tuple, List, Iterable

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib import rcParams
from matplotlib.axes import Axes
from pyquaternion import Quaternion
from tqdm import tqdm


from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from nuscenes.utils.map_mask import MapMask
from nuscenes.utils.color_map import get_colormap

from torchvision import transforms as T
from open3d import geometry
import copy


def get_projected_rgb_pointcloud(pointsensor_token: str, camera_token: str, nusc, idx):

    cam = nusc.get('sample_data', camera_token)
    pointsensor = nusc.get('sample_data', pointsensor_token)

    pc = LidarPointCloud.from_file(osp.join(nusc.dataroot, pointsensor['filename']))
    im = Image.open(osp.join(nusc.dataroot, cam['filename']))


    
    
    # transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.

    cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))


    # transform from ego to the global frame.
    poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    pc.translate(np.array(poserecord['translation']))

    # transform from global into the ego vehicle frame for the timestamp of the image.
    poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
    pc.translate(-np.array(poserecord['translation']))
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)
    
    cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    pc.translate(-np.array(cs_record['translation']))
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

    depths = pc.points[2, :]

    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)

    # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > 0)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
    points = points[:, mask]

    im_data = np.array(im)
    coloring = np.zeros(points.shape)
    
    for i, p in enumerate(points.transpose()):
        point = p[:2].round().astype(np.int32)
        coloring[:, i] = im_data[point[1], point[0], :]
    
    
    RGB = np.zeros_like(pc.points[0:3,:], dtype=np.uint8)
    XYZ = copy.deepcopy(pc.points)
    RGB[:,mask] = coloring
    
    xyzrgb_data=np.vstack((XYZ, RGB))
    xyzrgb_data=xyzrgb_data.transpose(1,0)

    RGB=RGB.astype(float).transpose()
    
    if not ((RGB >= 0.0) & (RGB <= 1.0)).all():
         RGB /= 255.0


    points=points.transpose(1,0)
    coloring=coloring.transpose(1,0)
    
    return RGB, xyzrgb_data, points, coloring, mask



