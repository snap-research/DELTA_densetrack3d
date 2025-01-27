"""Record3D visualizer

Parse and stream record3d captures. To get the demo data, see `./assets/download_record3d_dance.sh`.
"""

import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pickle
import time
from pathlib import Path

import cv2
import imageio
import mediapy as media
import numpy as onp
import numpy as np
import tyro
import viser
import viser.extras
import viser.transforms as tf
from densetrack3d.datasets.custom_data import read_data
from tqdm.auto import tqdm


def main(
    filepath: str = "results/demo/rollerblade/dense_3d_track.pkl",
    downsample_factor: int = 1,
    max_frames: int = 100,
    share: bool = True,
    port: int = 8080,
) -> None:
    server = viser.ViserServer(port=port)
    if share:
        server.request_share_url()

    print("Loading frames!")

    # trajs_data = np.load(data_path)
    with open(filepath, "rb") as handle:
        trajs_3d_dict = pickle.load(handle)

    coords = trajs_3d_dict["coords"].astype(np.float32)  # T N 3
    colors = trajs_3d_dict["colors"].astype(np.float32) / 255.0  # N 3
    vis = trajs_3d_dict["vis"].astype(np.float32)  # T N
    # trajs = trajs_data[:, :, :3] # T N 3
    # trajs[..., :2] *= trajs[..., 2:3]

    num_frames, num_points = coords.shape[:2]
    print(f"Num frames {num_frames}, Num points {num_points}")

    # breakpoint()

    filename = os.path.basename(filepath).split(".")[0]
    try:
        video, videodepth = read_data("demo_data", filename)
    except:
        video, videodepth = None, None
    # Add playback UI.
    with server.gui.add_folder("Playback"):
        gui_timestep = server.gui.add_slider(
            "Timestep",
            min=0,
            max=num_frames - 1,
            step=1,
            initial_value=0,
            disabled=True,
        )
        gui_next_frame = server.gui.add_button("Next Frame", disabled=True)
        gui_prev_frame = server.gui.add_button("Prev Frame", disabled=True)
        gui_playing = server.gui.add_checkbox("Playing", True)
        gui_framerate = server.gui.add_slider("FPS", min=1, max=60, step=0.1, initial_value=12)
        gui_framerate_options = server.gui.add_button_group("FPS options", ("10", "20", "30", "60"))

    # Frame step buttons.
    @gui_next_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value + 1) % num_frames

    @gui_prev_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value - 1) % num_frames

    # Disable frame controls when we're playing.
    @gui_playing.on_update
    def _(_) -> None:
        gui_timestep.disabled = gui_playing.value
        gui_next_frame.disabled = gui_playing.value
        gui_prev_frame.disabled = gui_playing.value

    # Set the framerate when we click one of the options.
    @gui_framerate_options.on_click
    def _(_) -> None:
        gui_framerate.value = int(gui_framerate_options.value)

    prev_timestep = gui_timestep.value

    # Toggle frame visibility when the timestep slider changes.
    @gui_timestep.on_update
    def _(_) -> None:
        nonlocal prev_timestep
        current_timestep = gui_timestep.value
        with server.atomic():
            frame_nodes[current_timestep].visible = True
            frame_nodes[prev_timestep].visible = False
        prev_timestep = current_timestep
        server.flush()  # Optional!

    # Load in frames.
    server.scene.add_frame(
        "/frames",
        wxyz=(1.0, 0.0, 0.0, 0.0),
        position=(0, 0, 0),
        show_axes=False,
    )

    frame_nodes: list[viser.FrameHandle] = []
    for i in tqdm(range(num_frames)):
        frame_nodes.append(server.scene.add_frame(f"/frames/t{i}", show_axes=False))

        # Place the point cloud in the frame.
        server.scene.add_point_cloud(
            name=f"/frames/t{i}/pos",
            points=coords[i],
            colors=colors,
            point_size=0.01,
            point_shape="rounded",
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(0.0, 0, 0),
        )

        if video is not None:
            img_i = video[i]
            img_h, img_w = img_i.shape[:2]
            # Place the frustum.
            fov = 2 * onp.arctan2(img_h / 2, img_w)
            aspect = img_w / img_h

            server.scene.add_camera_frustum(
                f"/frames/t{i}/frustum",
                fov=fov,
                aspect=aspect,
                scale=0.5,
                image=img_i,
                wxyz=(1.0, 0.0, 0.0, 0.0),
                position=(0.0, 0.0, -2.0),
            )

    # Hide all but the current frame.
    for i, frame_node in enumerate(frame_nodes):
        frame_node.visible = i == gui_timestep.value

    # Playback update loop.
    prev_timestep = gui_timestep.value
    while True:
        if gui_playing.value:
            gui_timestep.value = (gui_timestep.value + 1) % num_frames

        time.sleep(1.0 / gui_framerate.value)


if __name__ == "__main__":
    tyro.cli(main)
