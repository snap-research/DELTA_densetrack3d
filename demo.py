import os
import argparse
import pickle
import mediapy as media
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from densetrack3d.datasets.custom_data import read_data, read_data_with_depthcrafter
from densetrack3d.models.densetrack3d.densetrack3d import DenseTrack3D
from densetrack3d.models.geometry_utils import least_square_align
from densetrack3d.models.predictor.dense_predictor import DensePredictor3D
from densetrack3d.utils.depthcrafter_utils import read_video
from densetrack3d.utils.visualizer import Visualizer


BASE_DIR = os.getcwd()
device = torch.device("cuda")

@torch.inference_mode()
def predict_unidepth(video, model):
    video_torch = torch.from_numpy(video).permute(0, 3, 1, 2).to(device)

    depth_pred = []
    chunks = torch.split(video_torch, 32, dim=0)
    for chunk in chunks:
        predictions = model.infer(chunk)
        depth_pred_ = predictions["depth"].squeeze(1).cpu().numpy()
        depth_pred.append(depth_pred_)
    depth_pred = np.concatenate(depth_pred, axis=0)

    return depth_pred


@torch.inference_mode()
def predict_depthcrafter(video, pipe):
    frames, ori_h, ori_w = read_video(video, max_res=1024)
    res = pipe(
        frames,
        height=frames.shape[1],
        width=frames.shape[2],
        output_type="np",
        guidance_scale=1.2,
        num_inference_steps=25,
        window_size=110,
        overlap=25,
        track_time=False,
    ).frames[0]

    # convert the three-channel output to a single channel depth map
    res = res.sum(-1) / res.shape[-1]
    # normalize the depth map to [0, 1] across the whole video
    res = (res - res.min()) / (res.max() - res.min())

    res = F.interpolate(torch.from_numpy(res[:, None]), (ori_h, ori_w), mode="nearest").squeeze(1).numpy()

    return res


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="checkpoints/densetrack3d.pth", help="checkpoint path")
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
    model = DenseTrack3D(
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

    predictor = DensePredictor3D(model=model)
    predictor = predictor.eval().cuda()

    video, videodepth, videodisp = read_data_with_depthcrafter(full_path=args.video_path)

    if videodepth is None:

        os.sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "submodules", "UniDepth")))
        from unidepth.models import UniDepthV2
        from unidepth.utils import colorize, image_grid

        
        unidepth_model = UniDepthV2.from_pretrained(f"lpiccinelli/unidepth-v2-vitl14")
        unidepth_model = unidepth_model.eval().to(device)

        print("Run Unidepth")
        videodepth = predict_unidepth(video, unidepth_model)
        np.save(os.path.join(args.video_path, "depth_pred.npy"), videodepth)

    if args.use_depthcrafter:
        if videodisp is None:
            os.sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "submodules", "DepthCrafter")))
            from depthcrafter.depth_crafter_ppl import DepthCrafterPipeline
            from depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter
            from diffusers.training_utils import set_seed

            unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
                "tencent/DepthCrafter",
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
            )
            # load weights of other components from the provided checkpoint
            depth_crafter_pipe = DepthCrafterPipeline.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid-xt",
                unet=unet,
                torch_dtype=torch.float16,
                variant="fp16",
            )

            depth_crafter_pipe.to("cuda")
            # enable attention slicing and xformers memory efficient attention
            try:
                depth_crafter_pipe.enable_xformers_memory_efficient_attention()
            except Exception as e:
                print(e)
                print("Xformers is not enabled")
            depth_crafter_pipe.enable_attention_slicing()

            print("Run DepthCrafter")
            videodisp = predict_depthcrafter(video, depth_crafter_pipe)
            np.save(os.path.join(args.video_path, "depth_depthcrafter.npy"), videodisp)

        videodepth = least_square_align(videodepth, videodisp)

    video = torch.from_numpy(video).permute(0, 3, 1, 2).cuda()[None].float()
    videodepth = torch.from_numpy(videodepth).unsqueeze(1).cuda()[None].float()

    vid_name = args.video_path.split("/")[-1]
    save_dir = os.path.join(args.output_path, vid_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Save results to {save_dir}")

    print("Run DenseTrack3D")

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.use_fp16):
        out_dict = predictor(
            video,
            videodepth,
            grid_query_frame=0,
        )

    trajs_3d_dict = {k: v[0].cpu().numpy() for k, v in out_dict["trajs_3d_dict"].items()}

    
    with open(os.path.join(save_dir, f"dense_3d_track.pkl"), "wb") as handle:
        pickle.dump(trajs_3d_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

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
