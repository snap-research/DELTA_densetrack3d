import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import glob

import pickle
import numpy as np

import torch
import torch.nn.functional as F

import argparse
import cv2
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# from dust3r.cloud_opt.optimizer import PointCloudOptimizer
from dust3r.cloud_opt.delta_optimizer import DeltaPointCloudOptimizer
from dust3r.utils.device import to_numpy
from dust3r.utils.image import load_images, rgb, enlarge_seg_masks
from dust3r.io_utils import save_colmap_cameras, save_colmap_images, storePly
from densetrack3d.datasets.custom_data import read_data, read_data_with_depthcrafter

from sfm.sfm_utils import make_pairs, get_3D_model_from_scene



def prepare_single_pair(pair, pair_id, data_dict, keyframe_id_list, keyframe_masks, view1, view2, pred1, pred2, resolution=(384,512)):
    view1['img'].append(pair[0])
    view2['img'].append(pair[1])
    view1['idx'].append(pair_id[0])
    view2['idx'].append(pair_id[1])
    view1['instance'].append(str(pair_id[0]))
    view2['instance'].append(str(pair_id[1]))
    view1['true_shape'].append(torch.tensor(resolution))
    view2['true_shape'].append(torch.tensor(resolution))

    # pred1['pts3d'].append(data_dict[str(pair_id[0])]['tracks_3d'])

    keyframe1 = str(keyframe_id_list[pair_id[0]])
    keyframe2 = str(keyframe_id_list[pair_id[1]])

    mask1 = keyframe_masks[pair_id[0]]
    mask2 = keyframe_masks[pair_id[1]]

    view1['dynamic_mask'].append(mask1.float())
    view2['dynamic_mask'].append(mask2.float())

    pts3d_1 = data_dict[keyframe1]['trajs_3d'][keyframe1][:,:3].astype(np.float32)
    conf_1 = data_dict[keyframe1]['conf'][keyframe2].astype(np.float32) # NOTE trick here, conf of view1 is the visibility at view 2
    pts3d_1 = torch.from_numpy(pts3d_1).float().reshape(*resolution,3)
    conf_1 = torch.from_numpy(conf_1).float().reshape(*resolution)

    conf_1 = 1 + conf_1.exp()
    pts3d_2_at_1 = data_dict[keyframe2]['trajs_3d'][keyframe1][:,:3].astype(np.float32)


    conf_2_at_1 = data_dict[keyframe2]['conf'][keyframe1].astype(np.float32)
    pts3d_2_at_1 = torch.from_numpy(pts3d_2_at_1).float().reshape(*resolution,3)
    conf_2_at_1 = torch.from_numpy(conf_2_at_1).float().reshape(*resolution)
    conf_2_at_1 = 1.0 + conf_2_at_1.exp()

    pts2d_1 = data_dict[keyframe1]['trajs_uv'][keyframe1][:,:2].astype(np.float32)
    pts2d_1_at_2 = data_dict[keyframe1]['trajs_uv'][keyframe2][:,:2].astype(np.float32)

    pts2d_1 = torch.from_numpy(pts2d_1).float().reshape(*resolution,2)
    pts2d_1_at_2 = torch.from_numpy(pts2d_1_at_2).float().reshape(*resolution,2)
    flow2d_12 = pts2d_1_at_2 - pts2d_1

    visib_1_at_2 = data_dict[keyframe1]['conf'][keyframe2].astype(np.float32)
    visib_1_at_2 = (torch.from_numpy(visib_1_at_2) > 0.9).float().reshape(*resolution)

    pts2d_2 = data_dict[keyframe2]['trajs_uv'][keyframe2][:,:2].astype(np.float32)
    pts2d_2_at_1 = data_dict[keyframe2]['trajs_uv'][keyframe1][:,:2].astype(np.float32)

    pts2d_2 = torch.from_numpy(pts2d_2).float().reshape(*resolution,2)
    pts2d_2_at_1 = torch.from_numpy(pts2d_2_at_1).float().reshape(*resolution,2)
    flow2d_21 = pts2d_2_at_1 - pts2d_2

    visib_2_at_1 = data_dict[keyframe2]['conf'][keyframe1].astype(np.float32)
    visib_2_at_1 = (torch.from_numpy(visib_2_at_1) > 0.9).float().reshape(*resolution)

    pred1['pts3d'].append(pts3d_1)
    pred1['conf'].append(conf_1)
    pred1['flow2d'].append(flow2d_12)
    pred1['visib'].append(visib_1_at_2)

    pred2['pts3d_in_other_view'].append(pts3d_2_at_1)
    pred2['conf'].append(conf_2_at_1)
    pred2['flow2d'].append(flow2d_21)
    pred2['visib'].append(visib_2_at_1)


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default="demo_data/rollerblade", help="demo video path")
    parser.add_argument("--output_path", type=str, default="results/demo", help="output path")


    return parser

if __name__ == "__main__":

    parser = get_args_parser()
    args = parser.parse_args()

    vid_name = args.video_path.split("/")[-1]
    save_dir = os.path.join(args.output_path, vid_name)
    assert os.path.exists(save_dir)




    data_path = os.path.join(save_dir, "pred_dict_pairwise.pkl")
    with open(data_path, 'rb') as handle:
        data_dict = pickle.load(handle)

    video, _, _ = read_data_with_depthcrafter(full_path=args.video_path)

    
    # NOTE use GROUNDED-SAM to automatically obtain dynamic/foreground masks from video, then save them to the video_path/mask folder
    if os.path.exists(os.path.join(args.video_path, "mask")):
        videomask_paths = sorted(glob.glob(os.path.join(args.video_path, "mask", "*.png")))
        videomask = []
        for videomask_path in videomask_paths:
            mask = cv2.imread(videomask_path, cv2.IMREAD_GRAYSCALE)
            videomask.append(mask)
        videomask = np.stack(videomask).astype(bool)
        
    else:
        videomask = np.zeros_like(video[...,0]).astype(bool)

    video = torch.from_numpy(video).permute(0,3,1,2)[None].float()
    videomask = torch.from_numpy(videomask).unsqueeze(1)[None].bool()

    resized_hw = data_dict['0']['conf']['0'].shape[0]
    resized_w = 512
    assert resized_hw % resized_w == 0
    resized_h = int(resized_hw / 512 )

    resolution = (resized_h, resized_w)
    if video.shape[-2] != resolution[0] or video.shape[-1] != resolution[1]:
        B, T = video.shape[:2]
        video = F.interpolate(video.flatten(0,1), size=resolution, mode='bilinear', align_corners=False).reshape(B,T,3,*resolution)
        if videomask is not None:
            videomask = F.interpolate(videomask.float().flatten(0,1), size=resolution, mode='nearest').reshape(B,T,1,*resolution).bool()


    start_f = 0
    end_f = 60
    total_keyframe_id_list = list(data_dict.keys())[start_f:end_f]
    total_keyframes = len(total_keyframe_id_list)
    keyframe_id_list = total_keyframe_id_list
    keyframe_id_list = [int(k) for k in keyframe_id_list]

    print("keyframe_id_list", keyframe_id_list)

    keyframe_imgs = video[0, keyframe_id_list]
    keyframe_imgs = (keyframe_imgs / 255.0) * 2.0 - 1.0

    keyframe_masks = videomask[0, keyframe_id_list]
    keyframe_masks = keyframe_masks.squeeze(1)

    num_keyframes = len(keyframe_id_list)
    pairs_id = make_pairs(keyframe_imgs)

    pairs = []
    for (i,j) in pairs_id:
        pairs.append((keyframe_imgs[i], keyframe_imgs[j]))

    print("Num pairs:", len(pairs))

    view1 = {
        'img': [],
        'dynamic_mask': [],
        'true_shape': [],
        'idx': [],
        'instance': []
    }

    view2 = {
        'img': [],
        'dynamic_mask': [],
        'true_shape': [],
        'idx': [],
        'instance': []
    }

    pred1 = {
        'pts3d': [],
        'conf': [],
        # 'flow3d': [],
        'flow2d': [],
        'flow2d': [],
        "visib": []
    }

    pred2 = {
        'pts3d_in_other_view': [],
        'conf': [],
        # 'flow3d': [],
        'flow2d': [],
        "visib": []
    }

    # breakpoint()

    for pair, pair_id in zip(pairs, pairs_id):
        prepare_single_pair(pair, pair_id, data_dict, keyframe_id_list, keyframe_masks, view1, view2, pred1, pred2, resolution=resolution)

    view1['img'] = torch.stack(view1['img']).cuda()
    view2['img'] = torch.stack(view2['img']).cuda()
    view1['dynamic_mask'] = torch.stack(view1['dynamic_mask']).cuda()
    view2['dynamic_mask'] = torch.stack(view2['dynamic_mask']).cuda()
    view1['true_shape'] = torch.stack(view1['true_shape'], dim=0).cuda()
    view2['true_shape'] = torch.stack(view2['true_shape'], dim=0).cuda()

    pred1['pts3d'] = torch.stack(pred1['pts3d']).cuda()
    pred1['conf'] = torch.stack(pred1['conf']).cuda()
    pred1['flow2d'] = torch.stack(pred1['flow2d']).cuda()
    pred1['visib'] = torch.stack(pred1['visib']).cuda()

    pred2['pts3d_in_other_view'] = torch.stack(pred2['pts3d_in_other_view']).cuda()
    pred2['conf'] = torch.stack(pred2['conf']).cuda()
    pred2['flow2d'] = torch.stack(pred2['flow2d']).cuda()
    pred2['visib'] = torch.stack(pred2['visib']).cuda()

    device ="cuda"

    num_total_iter = 300
    schedule = "linear"
    lr = 0.01
    log_dir = None
    # os.makedirs(log_dir, exist_ok=True)
    optim_kw = {
        'verbose': True, 
        'min_conf_thr': 0.5, 
        'log_dir': log_dir, 
        'num_total_iter': num_total_iter, 
        'use_self_dynamic_mask': False
    }
    scene = DeltaPointCloudOptimizer(view1, view2, pred1, pred2, **optim_kw).to(device)

    loss = scene.compute_global_alignment(
        init='mst', 
        niter=num_total_iter, 
        schedule=schedule, 
        lr=lr,
        use_delta_init=True,
    )
                   
    # NOTE second stage

    # scene.freeze_pose()
    # loss = scene.compute_global_alignment(
    #     init=None, 
    #     niter=200, 
    #     schedule=schedule, 
    #     lr=0.01,
    #     stage='2'
    # )



    save_folder = f'vis_results/custom_data/global_alignment_delta/{vid_name}'  #default is 'demo_tmp/NULL'
    os.makedirs(save_folder, exist_ok=True)
    outfile = get_3D_model_from_scene(
        save_folder, 
        silent=True, 
        scene=scene, 
        min_conf_thr=0.9, 
        as_pointcloud=True, 
        mask_sky=False,
        clean_depth=True, 
        transparent_cams=False, 
        cam_size=0.05, 
        show_cam=True,
    )

    poses = scene.save_tum_poses(f'{save_folder}/pred_traj.txt')
    K = scene.save_intrinsics(f'{save_folder}/pred_intrinsics.txt')
    depth_maps = scene.save_depth_maps(save_folder)
    dynamic_masks = scene.save_dynamic_masks(save_folder)
    conf = scene.save_conf_maps(save_folder)
    init_conf = scene.save_init_conf_maps(save_folder)
    rgbs = scene.save_rgb_imgs(save_folder)
    enlarge_seg_masks(save_folder, kernel_size=3) 

    dynamic_masks = to_numpy(dynamic_masks)

    output_colmap_path = os.path.join(save_folder, "sparse/0")
    os.makedirs(output_colmap_path, exist_ok=True)

    imgs = to_numpy(scene.imgs)
    focals = scene.get_focals()
    poses = to_numpy(scene.get_im_poses())
    pts3d, _ = scene.get_pts3d(return_pose=True, fuse=True)
    pts3d = to_numpy(pts3d)
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(0.0)))
    confidence_masks = to_numpy(scene.get_masks())
    intrinsics = to_numpy(scene.get_intrinsics())

    # save
    save_colmap_cameras((resolution[1],resolution[0]), intrinsics, os.path.join(output_colmap_path, 'cameras.txt'))
    save_colmap_images(poses, os.path.join(output_colmap_path, 'images.txt'), keyframe_id_list)

    # storePly(os.path.join(output_colmap_path, "points3D_static.ply"), pts_static_3dgs, color_static_3dgs)

    pts_4_3dgs = np.concatenate([p.reshape(-1, 3) for p in pts3d])
    color_4_3dgs = np.concatenate([p.reshape(-1, 3) for p in imgs])
    color_4_3dgs = (color_4_3dgs * 255.0).astype(np.uint8)

    # breakpoint()
    storePly(os.path.join(output_colmap_path, "points3D.ply"), pts_4_3dgs, color_4_3dgs)
    pts_4_3dgs_all = np.array(pts3d).reshape(-1, 3)
    np.save(output_colmap_path + "/pts_4_3dgs_all.npy", pts_4_3dgs_all)
    np.save(output_colmap_path + "/focal.npy", np.array(focals.detach().cpu()))
    
