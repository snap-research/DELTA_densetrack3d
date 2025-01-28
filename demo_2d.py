import os
import argparse
import pickle
import mediapy as media
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from densetrack3d.datasets.custom_data import read_data
from densetrack3d.models.densetrack3d.densetrack2d import DenseTrack2D
from densetrack3d.models.predictor.dense_predictor2d import DensePredictor2D
from densetrack3d.utils.visualizer import Visualizer


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="checkpoints/densetrack2d.pth", help="checkpoint path")
    parser.add_argument("--video_path", type=str, default="demo_data/rollerblade", help="demo video path")
    parser.add_argument("--output_path", type=str, default="results/demo", help="output path")
    parser.add_argument(
        "--use_depthcrafter", action="store_true", help="whether to use depthcrafter as input videodepth"
    )
    parser.add_argument("--viz_sparse", type=bool, default=True, help="whether to viz sparse tracking")
    parser.add_argument("--downsample", type=int, default=16, help="downsample factor of sparse tracking")
    parser.add_argument("--upsample_factor", type=int, default=4, help="model stride")
    parser.add_argument("--use_fp16", action="store_true", help="whether to use fp16 precision")

    return parser


if __name__ == "__main__":

    parser = get_args_parser()
    args = parser.parse_args()

    print("Create DenseTrack3D model")
    model = DenseTrack2D(
        stride=4,
        window_len=16,
        add_space_attn=True,
        num_virtual_tracks=64,
        model_resolution=(384, 512),
        upsample_factor=args.upsample_factor
    )

    print(f"Load checkpoint from {args.ckpt}")
    with open(args.ckpt, "rb") as f:
        state_dict = torch.load(f, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
    model.load_state_dict(state_dict, strict=False)

    predictor = DensePredictor2D(model=model)
    predictor = predictor.eval().cuda()

    video, _ = read_data(full_path=args.video_path)

    video = torch.from_numpy(video).permute(0, 3, 1, 2).cuda()[None].float()

    vid_name = args.video_path.split("/")[-1]
    save_dir = os.path.join(args.output_path, vid_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Save results to {save_dir}")

    print("Run DenseTrack2D")

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.use_fp16):
        out_dict = predictor(
            video,
            grid_query_frame=0,
        )

    trajs_2d_dict = {
        "coords": out_dict["trajs_uv"][0].cpu().numpy(),
        "vis": out_dict["vis"][0].cpu().numpy(),
        "conf": out_dict["conf"][0].cpu().numpy()
    }

    
    with open(os.path.join(save_dir, f"dense_2d_track.pkl"), "wb") as handle:
        pickle.dump(trajs_2d_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if args.viz_sparse:
        print("Visualize sparse 2D tracking")
        W = video.shape[-1]
        visualizer_2d = Visualizer(
            save_dir="results/demo", fps=10, show_first_frame=0, linewidth=int(1 * W / 512), tracks_leave_trace=10
        )

        trajs_uv = out_dict["trajs_uv"]
        trajs_vis = out_dict["vis"]
        dense_reso = out_dict["dense_reso"]

        sparse_trajs_uv = rearrange(trajs_uv, "b t (h w) c -> b t h w c", h=dense_reso[0], w=dense_reso[1])
        sparse_trajs_uv = sparse_trajs_uv[:, :, :: args.downsample, :: args.downsample]
        sparse_trajs_uv = rearrange(sparse_trajs_uv, "b t h w c -> b t (h w) c")

        sparse_trajs_vis = rearrange(trajs_vis, "b t (h w) -> b t h w", h=dense_reso[0], w=dense_reso[1])
        sparse_trajs_vis = sparse_trajs_vis[:, :, :: args.downsample, :: args.downsample]
        sparse_trajs_vis = rearrange(sparse_trajs_vis, "b t h w -> b t (h w)")

        video2d_viz = visualizer_2d.visualize(
            video, sparse_trajs_uv, sparse_trajs_vis[..., None], filename="demo", save_video=False
        )

        video2d_viz = video2d_viz[0].permute(0, 2, 3, 1).cpu().numpy()
        media.write_video(os.path.join(save_dir, f"sparse_2d_track.mp4"), video2d_viz, fps=10)
