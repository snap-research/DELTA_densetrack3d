from scipy.spatial.transform import Rotation
import numpy as np
import trimesh
from monst3r.utils.device import to_numpy
import torch
import os
import cv2
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from third_party.raft import load_RAFT
# from datasets_preprocess.sintel_get_dynamics import compute_optical_flow
from dust3r.utils.flow_vis import flow_to_image

def depth_to_3d(depth_map, intrinsic_matrix):
    height, width = depth_map.shape
    i, j = np.meshgrid(np.arange(width), np.arange(height))
    
    # Convert pixel coordinates and depth values to 3D points
    x = (i - intrinsic_matrix[0, 2]) * depth_map / intrinsic_matrix[0, 0]
    y = (j - intrinsic_matrix[1, 2]) * depth_map / intrinsic_matrix[1, 1]
    z = depth_map
    
    points_3d = np.stack([x, y, z], axis=-1)
    return points_3d

def project_3d_to_2d(points_3d, intrinsic_matrix):
    # Convert 3D points to homogeneous coordinates
    projected_2d_hom = intrinsic_matrix @ points_3d.T
    # Convert from homogeneous coordinates to 2D image coordinates
    projected_2d = projected_2d_hom[:2, :] / projected_2d_hom[2, :]
    return projected_2d.T

def compute_optical_flow(depth1, depth2, pose1, pose2, intrinsic_matrix1, intrinsic_matrix2):
    # Input: All inputs as numpy arrays; convert torch tensors to numpy arrays if needed
    if isinstance(depth1, torch.Tensor):
        depth1 = depth1.cpu().numpy()
    if isinstance(depth2, torch.Tensor):
        depth2 = depth2.cpu().numpy()
    if isinstance(pose1, torch.Tensor):
        pose1 = pose1.cpu().numpy()
    if isinstance(pose2, torch.Tensor):
        pose2 = pose2.cpu().numpy()
    if isinstance(intrinsic_matrix1, torch.Tensor):
        intrinsic_matrix1 = intrinsic_matrix1.cpu().numpy()
    if isinstance(intrinsic_matrix2, torch.Tensor):
        intrinsic_matrix2 = intrinsic_matrix2.cpu().numpy()

    points_3d_frame1 = depth_to_3d(depth1, intrinsic_matrix1).reshape(-1, 3)
    points_3d_frame1_hom = np.concatenate([points_3d_frame1, np.ones((points_3d_frame1.shape[0], 1))], axis=1).T
    
    # Calculate the transformation matrix from frame 1 to frame 2
    transformation_matrix = (pose2) @ np.linalg.inv(pose1)
    points_3d_frame2_hom = transformation_matrix @ points_3d_frame1_hom
    points_3d_frame2 = (points_3d_frame2_hom[:3, :]).T

    points_2d_frame1 = project_3d_to_2d(points_3d_frame1, intrinsic_matrix1)
    points_2d_frame2 = project_3d_to_2d(points_3d_frame2, intrinsic_matrix2)

    # Compute optical flow vectors
    optical_flow = points_2d_frame2 - points_2d_frame1
    return optical_flow

def convert_scene_output_to_glb(outdir, imgs, pts3d, mask, focals, cams2world, cam_size=0.05, show_cam=True,
                                 cam_color=None, as_pointcloud=False,
                                 transparent_cams=False, silent=False, save_name=None):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()

    # full pointcloud

    # breakpoint()
    if as_pointcloud:
        pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
        pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d[i], mask[i]))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    # add each camera
    if show_cam:
        for i, pose_c2w in enumerate(cams2world):
            if isinstance(cam_color, list):
                camera_edge_color = cam_color[i]
            else:
                camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
            add_scene_cam(scene, pose_c2w, camera_edge_color,
                        None if transparent_cams else imgs[i], focals[i],
                        imsize=imgs[i].shape[1::-1], screen_width=cam_size)

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    if save_name is None: save_name='scene'
    outfile = os.path.join(outdir, save_name+'.glb')
    if not silent:
        print('(exporting 3D scene to', outfile, ')')
    scene.export(file_obj=outfile)
    return outfile

def get_dynamic_mask_from_pairviewer(scene, flow_net=None, both_directions=False, output_dir='./demo_tmp'):
    """
    get the dynamic mask from the pairviewer
    """
    if flow_net is None:
        # flow_net = load_RAFT(model_path="third_party/RAFT/models/Tartan-C-T-TSKH-spring540x960-M.pth").to('cuda').eval() # sea-raft
        flow_net = load_RAFT(model_path="third_party/RAFT/models/raft-things.pth").to('cuda').eval()

    imgs = scene.imgs
    img1 = torch.from_numpy(imgs[0]*255).permute(2,0,1)[None]                       # (B, 3, H, W)
    img2 = torch.from_numpy(imgs[1]*255).permute(2,0,1)[None]
    with torch.no_grad():
        forward_flow = flow_net(img1.cuda(), img2.cuda(), iters=20, test_mode=True)[1]  # (B, 2, H, W)
        if both_directions:
            backward_flow = flow_net(img2.cuda(), img1.cuda(), iters=20, test_mode=True)[1]
            
    B, _, H, W = forward_flow.shape

    depth_map1 = scene.get_depthmaps()[0] # (H, W)
    depth_map2 = scene.get_depthmaps()[1]

    im_poses = scene.get_im_poses()
    cam1 = im_poses[0]                  # (4, 4)   cam2world
    cam2 = im_poses[1]
    extrinsics1 = torch.linalg.inv(cam1) # (4, 4)   world2cam
    extrinsics2 = torch.linalg.inv(cam2)

    intrinsics = scene.get_intrinsics()
    intrinsics_1 = intrinsics[0]        # (3, 3)
    intrinsics_2 = intrinsics[1]

    ego_flow_1_2 = compute_optical_flow(depth_map1, depth_map2, extrinsics1, extrinsics2, intrinsics_1, intrinsics_2) # (H*W, 2)
    ego_flow_1_2 = ego_flow_1_2.reshape(H, W, 2).transpose(2, 0, 1) # (2, H, W)

    error_map = np.linalg.norm(ego_flow_1_2 - forward_flow[0].cpu().numpy(), axis=0) # (H, W)

    error_map_normalized = (error_map - error_map.min()) / (error_map.max() - error_map.min())
    error_map_normalized_int = (error_map_normalized * 255).astype(np.uint8)
    if both_directions:
        ego_flow_2_1 = compute_optical_flow(depth_map2, depth_map1, extrinsics2, extrinsics1, intrinsics_2, intrinsics_1)
        ego_flow_2_1 = ego_flow_2_1.reshape(H, W, 2).transpose(2, 0, 1)
        error_map_2 = np.linalg.norm(ego_flow_2_1 - backward_flow[0].cpu().numpy(), axis=0)
        error_map_2_normalized = (error_map_2 - error_map_2.min()) / (error_map_2.max() - error_map_2.min())
        error_map_2_normalized = (error_map_2_normalized * 255).astype(np.uint8)
        cv2.imwrite(f'{output_dir}/dynamic_mask_bw.png', cv2.applyColorMap(error_map_2_normalized, cv2.COLORMAP_JET))
        np.save(f'{output_dir}/dynamic_mask_bw.npy', error_map_2)

        backward_flow = backward_flow[0].cpu().numpy().transpose(1, 2, 0)
        np.save(f'{output_dir}/backward_flow.npy', backward_flow)
        flow_img = flow_to_image(backward_flow)
        cv2.imwrite(f'{output_dir}/backward_flow.png', flow_img)

    cv2.imwrite(f'{output_dir}/dynamic_mask.png', cv2.applyColorMap(error_map_normalized_int, cv2.COLORMAP_JET))
    error_map_normalized_bin = (error_map_normalized > 0.3).astype(np.uint8)
    # save the binary mask
    cv2.imwrite(f'{output_dir}/dynamic_mask_binary.png', error_map_normalized_bin*255)
    # save the original one as npy file
    np.save(f'{output_dir}/dynamic_mask.npy', error_map)

    # also save the flow
    forward_flow = forward_flow[0].cpu().numpy().transpose(1, 2, 0)
    np.save(f'{output_dir}/forward_flow.npy', forward_flow)
    # save flow as image
    flow_img = flow_to_image(forward_flow)
    cv2.imwrite(f'{output_dir}/forward_flow.png', flow_img)

    return error_map