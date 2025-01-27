import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


import argparse

import torch
import numpy as np
from einops import rearrange


from PIL import Image
from tqdm import tqdm

from densetrack3d.datasets.cvo_dataset import CVO

BASE_DIR = os.getcwd()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
UINT16_MAX = 65535

intr = torch.tensor([[560, 0 , 256], 
                    [0  ,560, 256],
                    [0,  0,   1]])

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="./datasets/dot_datasets/kubric/cvo", help="path to tapvid3d dataset")
    parser.add_argument("--split", type=str, default="adt", help="tapvid3d split", choices=["clean", "final", "extended"])
    parser.add_argument("--use_zoedepth", type=bool, default=False, help="Whether to use ZoeDepth, otherwise use UniDepth")

    return parser

if __name__ == '__main__':


    parser = get_args_parser()
    args = parser.parse_args()


    if args.use_zoedepth:
        model = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True).to('cuda').eval()
    else:
        os.sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "submodules", "UniDepth")))
        from unidepth.models import UniDepthV2
        model = UniDepthV2.from_pretrained(f"lpiccinelli/unidepth-v2-vitl14").to('cuda').eval()



    depth_dataset_root = os.path.join(args.dataset_root, f"cvo_test_{args.split}_depth")
    test_dataset = CVO(data_root=args.dataset_root, split=args.split, debug=False) 

    index = 0
    while True:
        try:
            sample = test_dataset.sampler.sample(index)

            video = torch.from_numpy(sample["imgs"].copy())
            video_torch = rearrange(video, "h w (t c) -> t c h w", c=3).cuda()
            intrinsic_mat_ = intr.repeat(video_torch.shape[0], 1, 1).cuda()

            if args.use_zoedepth: # NOTE zoe
                for img in video:
                    depth_pred = model.infer_pil(Image.fromarray(img))
                    depth_preds.append(depth_pred)
                depth_preds = np.stack(depth_preds, axis=0).astype(np.float16)
            else: # NOTE unidepth
                depth_preds = []
                chunk_size = 32
                video_torch_chunk = torch.split(video_torch, chunk_size, dim=0)
                for chunk in video_torch_chunk:
                    predictions = model.infer(chunk, intrinsic_mat_)
                    depth_pred = predictions["depth"].squeeze(1).cpu().numpy()
                    depth_preds.append(depth_pred)

                depth_preds = np.concatenate(depth_preds, axis=0).astype(np.float16)


            np.save(os.path.join(depth_dataset_root, f"{index:05d}"), depth_preds)

            print("Processed scene", index)
            index += 1

        except:
            print("Finish")
            break
