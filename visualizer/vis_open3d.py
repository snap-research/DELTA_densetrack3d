import open3d as o3d

import numpy as np
import os
import cv2
import matplotlib
import matplotlib.cm
import time 
import argparse
import pickle

import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

import mediapy as media
import platform

isMacOS = (platform.system() == "Darwin")



global global_vidname

def read_data(args):
    with open(args.filepath, "rb") as handle:
        trajs_3d_dict = pickle.load(handle)

    coords = trajs_3d_dict["coords"].astype(np.float32)  # T N 3
    colors = trajs_3d_dict["colors"].astype(np.float32) # N 3, 0->255
    vis = trajs_3d_dict["vis"].astype(np.float32)  # T N

    return coords, vis, colors

def save_viewpoint(vis):
    global global_vidname

    view_control = vis.get_view_control()
    camera_params = view_control.convert_to_pinhole_camera_parameters()

    save_dir = "./results/viewpoints"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{global_vidname}.json")
    
    o3d.io.write_pinhole_camera_parameters(save_path, camera_params)
    print("Viewpoint saved to", save_path)
    return False  # Returning False ensures the visualizer continues to run



def get_viewpoint(args):
    traj, vis, color = read_data(args)

    # NOTE prepare a binary foreground mask to highlight moving object
    try:
        binary_mask_ori = cv2.imread(args.fg_mask_path, cv2.IMREAD_GRAYSCALE)
        binary_mask = cv2.resize(binary_mask_ori, (512, 384), cv2.INTER_NEAREST).astype(bool) # NOTE resize to the same number of tracking points, in this case I track every pixels of reso 384x512, thus resize to this
        binary_mask = binary_mask.reshape(-1)
    except:
        binary_mask = np.ones((384,512)).astype(bool).reshape(-1)

    cmap = plt.get_cmap('gist_rainbow')
    traj_len = binary_mask.sum()
    norm =  matplotlib.colors.Normalize(vmin=-traj_len*0.1, vmax=traj_len*1.1)
    rainbow_colors = np.asarray(cmap(norm(np.arange(traj_len)))[:, :3]) * 255.0

    blend_w = 0.3
    color[binary_mask,:] = color[binary_mask,:] * blend_w + (1-blend_w) * rainbow_colors # NOTE blending color to highlight moving obj, does not necessary

    list_geometry = []
    geometry = o3d.geometry.PointCloud()
    geometry.points =  o3d.utility.Vector3dVector(traj[0])
    geometry.colors = o3d.utility.Vector3dVector(color /255.0)

    list_geometry.append(geometry)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    for geo in list_geometry:
        vis.add_geometry(geo)

    vis.register_key_callback(ord("S"), save_viewpoint) # NOTE print S to save camera parameter in open3D to json file

    vis.poll_events()
    vis.get_render_option().point_size = 6
    vis.run()
    vis.destroy_window()

def load_viewpoint(vis, filename="default.json"):
    # Load the camera parameters from the JSON file
    camera_params = o3d.io.read_pinhole_camera_parameters(filename)
    
    # Apply the loaded camera parameters to the view control of the visualizer
    view_control = vis.get_view_control()
    view_control.convert_from_pinhole_camera_parameters(camera_params)
    
    print(f"Viewpoint loaded from {filename}")



def capture(args):

    camera_params = o3d.io.read_pinhole_camera_parameters(f"./results/viewpoints/{args.video_name}.json")
    intrinsic = camera_params.intrinsic
    window_h, window_w = intrinsic.height, intrinsic.width
    
    visualizer = o3d.visualization.VisualizerWithKeyCallback()
    visualizer.create_window(width=window_w, height=window_h)
    visualizer.get_render_option().point_size = 6

    view_control = visualizer.get_view_control()

    visualizer.poll_events()
    visualizer.update_renderer()

    saved_folder = os.path.join("results/open3d_capture", args.video_name)
    os.makedirs(saved_folder, exist_ok=True)
    os.makedirs(os.path.join(saved_folder, "color"), exist_ok=True)
    

    traj, vis, color = read_data(args)
    T, N = traj.shape[:2]
    
    try:
        binary_mask_ori = cv2.imread(args.fg_mask_path, cv2.IMREAD_GRAYSCALE)
        binary_mask = cv2.resize(binary_mask_ori, (512, 384), cv2.INTER_NEAREST).astype(bool) # NOTE resize to the same number of tracking points, in this case I track every pixels of reso 384x512, thus resize to this
        binary_mask = binary_mask.reshape(-1)
    except:
        binary_mask = np.ones((384,512)).astype(bool).reshape(-1)

    foreground_mask = binary_mask
    background_mask = ~binary_mask

    caro_mask = np.zeros((384, 512)).astype(bool) # NOTE subsample points to draw trajectories, otherwise drawing all trajectires are very hard to see
    caro_mask[::4,::4] = 1 
    binary_mask2 = foreground_mask & caro_mask.reshape(-1)

    caro_mask = np.zeros((384, 512)).astype(bool)
    caro_mask[::10,::10] = 1
    binary_mask3 = background_mask & caro_mask.reshape(-1)

    binary_mask4 = binary_mask2 | binary_mask3
    
    # # NOTE optional, blending color to highlight moving obj, does not necessary
    # cmap = plt.get_cmap('gist_rainbow')
    # traj_len = binary_mask.sum()
    # norm =  matplotlib.colors.Normalize(vmin=-traj_len*0.1, vmax=traj_len*1.1)
    # rainbow_colors = np.asarray(cmap(norm(np.arange(traj_len)))[:, :3]) * 255.0
    # blend_w = 0.3
    # color[binary_mask,:] = color[binary_mask,:] * blend_w + (1-blend_w) * rainbow_colors
    # #############################!SECTION

    list_geometry = []
    for t in range(T):
        if len(list_geometry) > 0:# NOTE remove visualization from previous frame
            for g in list_geometry: 
                visualizer.remove_geometry(g)
            list_geometry = []

        # NOTE draw point cloud
        vis_pc = o3d.geometry.PointCloud()
        vis_pc.points =  o3d.utility.Vector3dVector(traj[t])
        vis_pc.colors = o3d.utility.Vector3dVector(color /255.0)
        list_geometry.append(vis_pc)

        # NOTE draw trajectories
        diff_track = (traj[:t, background_mask] - traj[t:t+1, background_mask]).mean(1) # T , 3 # NOTE compensate for camera motion
        for i in range(max(1, 1), t):
            p1 = traj[i-1, binary_mask4] - diff_track[i-1] # - delta * (65-i) #   + np.array([2/40 * (i-1 + 1), 0, 0])[None]
            p2 = traj[i, binary_mask4]  - diff_track[i] # - delta * (65-i+1) # + np.array([2/40 * (i + 1), 0, 0])[None]

            n_pts = p1.shape[0]
            vertices = np.concatenate([p1, p2], 0)
            lines = np.stack([np.arange(n_pts), np.arange(n_pts)+n_pts], 1)

            cmap = plt.get_cmap('gist_rainbow')

            traj_len = len(p1)
            norm =  matplotlib.colors.Normalize(vmin=-traj_len*0.1, vmax=traj_len*1.1)
            line_colors = np.asarray(cmap(norm(np.arange(traj_len)))[:, :3])
            lineset = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(vertices),
                lines=o3d.utility.Vector2iVector(lines)
            )
            lineset.colors = o3d.utility.Vector3dVector(line_colors)

            list_geometry.append(lineset)

        for g in list_geometry:
            visualizer.add_geometry(g)

        view_control.convert_from_pinhole_camera_parameters(camera_params, allow_arbitrary=True )
        visualizer.poll_events()
        visualizer.update_renderer()
        visualizer.capture_screen_image(os.path.join(saved_folder, "color", f'{t:05}.png'))
        time.sleep(0.01)

    time.sleep(2.0)

    video = []
    for t in range(T):
        img = cv2.imread(os.path.join(saved_folder, "color", f'{t:05}.png'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        video.append(img)
    video = np.stack(video, axis=0)

    save_video_path = os.path.join(saved_folder, f"video.mp4")
    media.write_video(save_video_path, video, fps=10)
    print("Video saved to", save_video_path)

    visualizer.destroy_window()

def parse_args():
    parser = argparse.ArgumentParser(description="Vis 3D Tracks with open3d")
    parser.add_argument('--filepath', type=str, default="results/car-roundabout/dense_3d_track.pkl", help='Path to the tracking file')
    parser.add_argument('--mode', type=str, choices=['choose_viewpoint', 'capture'], default='choose_viewpoint')
    parser.add_argument('--fg_mask_path', default='demo_data/car-roundabout/car-roundabout_mask.png')
    parser.add_argument('--video_name', default='car-roundabout')
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    
    global_vidname = args.video_name
    # NOTE for get camera parameter
    # NOTE: run this block first, this will open an Open3D window, 
    # then choose the view you want to capture, 
    # then press S to save camera parameter,then close the window
    if args.mode == "choose_viewpoint":
        get_viewpoint(args)
    elif args.mode == "capture":
        # NOTE for record video of whole sequence
        capture(args)
    else:
        raise ValueError("Unknown mode")

    #####################################