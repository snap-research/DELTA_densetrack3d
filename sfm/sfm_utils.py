import os
import os.path as osp
from collections import OrderedDict
import io
from PIL import Image

import json
import os
from dataclasses import dataclass, field

import cv2
from einops import rearrange, reduce, einsum
import pickle
import hydra
import numpy as np
import mediapy as media

import torch
import torch.nn.functional as F

import matplotlib.pyplot as pl


from dust3r.cloud_opt.optimizer import PointCloudOptimizer
from dust3r.utils.post_process import estimate_focal_knowing_depth
from dust3r.utils.device import to_numpy
from dust3r.utils.geometry import xy_grid, geotrf
from dust3r.utils.image import load_images, rgb, enlarge_seg_masks
from dust3r.utils.viz_demo import convert_scene_output_to_glb, get_dynamic_mask_from_pairviewer


def make_pairs(imgs):
    start = 1
    stride = 1
    winsize = 5
    iscyclic = False

    pairsid = set()

    for i in range(len(imgs)):
        for j in range(start, stride*winsize + start, stride):
            idx = (i + j)
            if iscyclic:
                idx = idx % len(imgs)  # explicit loop closure
            if idx >= len(imgs):
                continue
            pairsid.add((i, idx) if i < idx else (idx, i))
    
    sym_pairsid = list()
    for pairid in pairsid:
        sym_pairsid.append((pairid[0], pairid[1]))
    
    for pairid in pairsid:
        sym_pairsid.append((pairid[1], pairid[0]))

    return sym_pairsid


def get_3D_model_from_scene(outdir, silent, scene, min_conf_thr=3, as_pointcloud=False, mask_sky=False,
                            clean_depth=False, transparent_cams=False, cam_size=0.05, show_cam=True, save_name=None, thr_for_init_conf=True):
    """
    extract 3D_model (glb file) from a reconstructed scene
    """
    if scene is None:
        return None
    # post processes
    if clean_depth:
        scene = scene.clean_pointcloud()
    if mask_sky:
        scene = scene.mask_sky()

    # get optimized values from scene
    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()
    # 3D pointcloud from depthmap, poses and intrinsics
    pts3d = to_numpy(scene.get_pts3d(raw=False))
    scene.min_conf_thr = min_conf_thr
    scene.thr_for_init_conf = thr_for_init_conf
    msk = to_numpy(scene.get_masks())
    cmap = pl.get_cmap('viridis')
    cam_color = [cmap(i/len(rgbimg))[:3] for i in range(len(rgbimg))]
    cam_color = [(255*c[0], 255*c[1], 255*c[2]) for c in cam_color]
    return convert_scene_output_to_glb(outdir, rgbimg, pts3d, msk, focals, cams2world, as_pointcloud=as_pointcloud,
                                        transparent_cams=transparent_cams, cam_size=cam_size, show_cam=show_cam, silent=silent, save_name=save_name,
                                        cam_color=cam_color)
