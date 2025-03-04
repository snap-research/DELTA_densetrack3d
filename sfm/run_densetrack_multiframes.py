import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import cv2
import json
import os
from dataclasses import dataclass, field

import hydra
import numpy as np
import pickle

import torch
from omegaconf import OmegaConf

from tqdm import tqdm

import torch.nn.functional as F

import argparse


from densetrack3d.models.geometry_utils import least_square_align
from densetrack3d.datasets.custom_data import read_data, read_data_with_depthcrafter
from densetrack3d.models.densetrack3d.densetrack3d import DenseTrack3D
from densetrack3d.models.predictor.dense_predictor import DensePredictor3D
from densetrack3d.utils.depthcrafter_utils import read_video

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

    return parser

if __name__ == "__main__":

    parser = get_args_parser()
    args = parser.parse_args()

    # NOTE force using DepthCrafter: we found that the depthcrafter depth is more accurate than the unidepth depth for sfm
    args.use_depthcrafter = True

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

    vid_name = args.video_path.split("/")[-1]
    save_dir = os.path.join(args.output_path, vid_name)
    os.makedirs(save_dir, exist_ok=True)

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


    max_T = min(100, video.shape[1])
    stride = 2
    chunk_size = 16

    



    video = torch.from_numpy(video).permute(0,3,1,2).cuda()[None].float()
    videodepth = torch.from_numpy(videodepth).unsqueeze(1).cuda()[None].float()

    

    key_inds = list(range(0, max_T, stride))

    video = video[:, :max_T]
    videodepth = videodepth[:, :max_T]
    total_dict = {}
    
    for ind in key_inds:
        print(f"Densely tracking points at frame {ind}")

        fw_dict, bw_dict = None, None
        if ind < video.shape[1]:
            fw_video = video[:, ind:min(ind+chunk_size, video.shape[1])]
            fw_videodepth = videodepth[:, ind:min(ind+chunk_size, video.shape[1])]
            fw_out_dict = predictor(
                fw_video,
                fw_videodepth,
                grid_query_frame=0,
            )

            fw_dict = {
                'trajs_3d': fw_out_dict["trajs_3d_dict"]["coords"][0].cpu().numpy(),
                'trajs_uv': fw_out_dict["trajs_uv"][0].cpu().numpy(),
                'conf': fw_out_dict["conf"][0].cpu().numpy(),
            }

        if ind > 0:
            bw_video = video[:, max(0, ind-chunk_size):ind+1].flip(1)
            bw_videodepth = videodepth[:, max(0, ind-chunk_size):ind+1].flip(1)
            bw_out_dict = predictor(
                bw_video,
                bw_videodepth,
                grid_query_frame=0,
            )

            bw_dict = {
                'trajs_3d': bw_out_dict["trajs_3d_dict"]["coords"][0].cpu().numpy(),
                'trajs_uv': bw_out_dict["trajs_uv"][0].cpu().numpy(),
                'conf': bw_out_dict["conf"][0].cpu().numpy(),
            }

        strided_save_dict = {
            'trajs_3d': {},
            'trajs_uv': {},
            'conf': {},
        }

        if fw_dict is not None:
            for k in fw_dict.keys():
                for tstep in range(0, chunk_size, stride): # 0,2,4,6
                    if tstep >= len(fw_dict[k]): break

                    strided_save_dict[k][str(ind+tstep)] = fw_dict[k][tstep]

        if bw_dict is not None:
            for k in bw_dict.keys():
                for tstep in range(stride, chunk_size, stride):
                    if tstep >= len(bw_dict[k]): break

                    strided_save_dict[k][str(ind-tstep)] = bw_dict[k][tstep]

        total_dict[str(ind)] = strided_save_dict

    with open(os.path.join(save_dir, 'pred_dict_pairwise.pkl'), 'wb') as handle:
        pickle.dump(total_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

