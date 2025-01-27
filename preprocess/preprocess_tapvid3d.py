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


from tqdm import tqdm

import ray

BASE_DIR = os.getcwd()



# TAPVID3D_ROOT = "<PATH_TO_TAPVID3D_DATASET>"
# TAPVID3D_SPLIT

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

@ray.remote(num_gpus=1)
class DepthPredictor:
    def __init__(self, worker_id, use_zoedepth=False):
        self.worker_id = worker_id
        self.use_zoedepth = use_zoedepth
        
        if use_zoedepth:
            self.model = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True).to('cuda').eval()
        else:
            os.sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "submodules", "UniDepth")))
            from unidepth.models import UniDepthV2
            self.model = UniDepthV2.from_pretrained(f"lpiccinelli/unidepth-v2-vitl14").to('cuda').eval()


        self.dataset_root = dataset_root

    def infer(self, seq_names, device='cuda'):
        for seq_name in tqdm(seq_names):

            gt_path = os.path.join(self.dataset_root, f"{seq_name}.npz")
                
            with open(gt_path, 'rb') as in_f:
                in_npz = np.load(in_f, allow_pickle=True)
                images_jpeg_bytes = in_npz['images_jpeg_bytes']
                video = []
                for frame_bytes in images_jpeg_bytes:
                    arr = np.frombuffer(frame_bytes, np.uint8)
                    image_bgr = cv2.imdecode(arr, flags=cv2.IMREAD_UNCHANGED)
                    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                    video.append(image_rgb)
                video = np.stack(video, axis=0)
                intrinsics_params = in_npz['fx_fy_cx_cy']

                intrinsic_mat = np.array([[intrinsics_params[0], 0, intrinsics_params[2]], [0, intrinsics_params[1], intrinsics_params[3]], [0, 0, 1]])
                intrinsic_mat = torch.from_numpy(intrinsic_mat).float().to(device)[None]

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
                        intrinsic_mat_ = intrinsic_mat.repeat(chunk.shape[0], 1, 1)
                        predictions = self.model.infer(chunk, intrinsic_mat_)
                        depth_pred = predictions["depth"].squeeze(1).cpu().numpy()
                        depth_preds.append(depth_pred)

                    depth_preds = np.concatenate(depth_preds, axis=0).astype(np.float16)


                out_npz = {}
                for k, v in in_npz.items():
                    out_npz[k] = v

                if self.use_zoedepth:
                    out_npz['depth_preds_zoe'] = depth_preds
                else:
                    out_npz['depth_preds'] = depth_preds

                np.savez_compressed(os.path.join(self.dataset_root, f"{seq_name}.npz"), **out_npz)

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="./datasets/tapvid3d", help="path to tapvid3d dataset")
    parser.add_argument("--split", type=str, default="adt", help="tapvid3d split", choices=["adt", "drivetrack", "pstudio"])
    parser.add_argument("--use_zoedepth", type=bool, default=False, help="Whether to use ZoeDepth, otherwise use UniDepth")

    return parser

if __name__ == '__main__':

    parser = get_args_parser()
    args = parser.parse_args()

    num_gpus = min(8, torch.cuda.device_count())

    print(f"Using {num_gpus} GPUs")

    dataset_root = os.path.join(args.dataset_root, args.split)
    assert os.path.isdir(dataset_root), f"Dataset root {dataset_root} does not exist"

    seq_names = sorted([f.split('.')[0] for f in os.listdir(dataset_root) if f.endswith(".npz")])

    chunks = split_chunks(seq_names, num_gpus)

    print("chunks", [len(c) for c in chunks])

    ray.init()
    detectors = [DepthPredictor.remote(i, use_zoedepth=args.use_zoedepth) for i in range(num_gpus)]
    tasks = [detector.infer.remote(chunk) for detector, chunk in zip(detectors, chunks)]

    ray.get(tasks)