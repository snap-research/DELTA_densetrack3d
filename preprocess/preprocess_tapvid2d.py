import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse

# import lmdb
# import pyarrow as pa
import torch
import numpy as np
import cv2
from PIL import Image

import io
from tqdm import tqdm
import pickle
import ray

BASE_DIR = os.getcwd()


def split_chunks(lst, M):
    """
    Split a list `lst` into `M` sublists with approximately equal numbers of elements.

    Args:
        lst (list): The list to split.
        M (int): The number of sublists.

    Returns:
        list of lists: A list containing `M` sublists.
    """
    # Calculate the length of each sublist and the remainder
    n = len(lst)
    avg = n // M
    remainder = n % M

    # Create the sublists
    chunks = []
    start = 0
    for i in range(M):
        end = start + avg + (1 if i < remainder else 0)
        chunks.append(lst[start:end])
        start = end

    return chunks

# @ray.remote(num_gpus=1)
class DepthPredictor:
    def __init__(self, worker_id, dataset_root, use_zoedepth=False, split="davis"):
        self.worker_id = worker_id
        self.use_zoedepth = use_zoedepth
        self.split = split
        
        if use_zoedepth:
            self.model = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True).to('cuda').eval()
        else:
            os.sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "submodules", "UniDepth")))
            from unidepth.models import UniDepthV2
            self.model = UniDepthV2.from_pretrained(f"lpiccinelli/unidepth-v2-vitl14").to('cuda').eval()


        self.dataset_root = dataset_root

    def infer(self, device='cuda'):
        
        with open(self.dataset_root, "rb") as f:
            points_dataset = pickle.load(f)

        if self.split == "davis":
            video_names = list(points_dataset.keys())
        else:
            video_names = list(range(len(points_dataset)))

        new_points_dataset = []

        # print(points_dataset.keys())
        for i, video_name in tqdm(enumerate(video_names)):
            sample = points_dataset[video_name].copy()
            video = sample["video"]

            if isinstance(video[0], bytes):
                # TAP-Vid is stored and JPEG bytes rather than `np.ndarray`s.
                def decode(frame):
                    byteio = io.BytesIO(frame)
                    img = Image.open(byteio)
                    return np.array(img)

                video = np.array([decode(frame) for frame in video])

            if self.use_zoedepth: # NOTE zoe
                for img in video:
                    depth_pred = self.model.infer_pil(Image.fromarray(img))
                    depth_preds.append(depth_pred)
                depth_preds = np.stack(depth_preds, axis=0).astype(np.float16)
            else: # NOTE unidepth
                depth_preds = []
                video_torch = torch.from_numpy(video).permute(0,3,1,2).to(device)
                chunk_size = 32
                video_torch_chunk = torch.split(video_torch, chunk_size, dim=0)
                for chunk in video_torch_chunk:
                    predictions = self.model.infer(chunk)
                    depth_pred = predictions["depth"].squeeze(1).cpu().numpy()
                    depth_preds.append(depth_pred)

                depth_preds = np.concatenate(depth_preds, axis=0).astype(np.float16)

            new_sample = {}
            for k, v in sample.items():
                new_sample[k] = v

            if self.use_zoedepth:
                new_sample['depth_preds_zoe'] = depth_preds
            else:
                new_sample['depth_preds'] = depth_preds

            new_points_dataset.append(new_sample)

        dir_name = os.path.dirname(self.dataset_root)
        base_name = os.path.basename(self.dataset_root)
        new_dataset_root = os.path.join(dir_name, base_name.replace(".pkl", "_with_depth.pkl"))
        with open(new_dataset_root, 'wb') as handle:
            pickle.dump(new_points_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="./datasets/tapvid2d", help="path to tapvid2d dataset")
    parser.add_argument("--split", type=str, default="adt", help="tapvid3d split", choices=["davis", "rgb_stacking", "kinetics"])
    parser.add_argument("--use_zoedepth", type=bool, default=False, help="Whether to use ZoeDepth, otherwise use UniDepth")

    return parser

if __name__ == '__main__':
    
    parser = get_args_parser()
    args = parser.parse_args()

    num_gpus = 1
    print(f"Using {num_gpus} GPUs")


    # NOTE double-check this
    dataset_root = os.path.join(args.dataset_root, f"tapvid_{args.split}", f"tapvid_{args.split}.pkl")
    assert os.path.exists(dataset_root), f"Dataset root {dataset_root} does not exist"

    detector = DepthPredictor(0, dataset_root=dataset_root, split=args.split)
    detector.infer()

        