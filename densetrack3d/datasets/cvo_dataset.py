import os
import os.path as osp
import pickle as pkl
from collections import OrderedDict

import lmdb
import numpy as np

# import pyarrow as pa
import torch
from densetrack3d.datasets.utils import DeltaData
from densetrack3d.models.model_utils import get_grid
from einops import rearrange
from torch.utils.data import DataLoader, Dataset


def get_alpha_consistency(bflow, fflow, thresh_1=0.01, thresh_2=0.5, thresh_mul=1):
    norm = lambda x: x.pow(2).sum(dim=-1).sqrt()
    B, H, W, C = bflow.shape

    mag = norm(fflow) + norm(bflow)
    grid = get_grid(H, W, shape=[B], device=fflow.device)
    grid[..., 0] = grid[..., 0] + bflow[..., 0] / (W - 1)
    grid[..., 1] = grid[..., 1] + bflow[..., 1] / (H - 1)
    grid = grid * 2 - 1
    fflow_warped = torch.nn.functional.grid_sample(
        fflow.permute(0, 3, 1, 2), grid, mode="bilinear", align_corners=True
    )
    flow_diff = bflow + fflow_warped.permute(0, 2, 3, 1)
    occ_thresh = thresh_1 * mag + thresh_2
    occ_thresh = occ_thresh * thresh_mul
    alpha = norm(flow_diff) < occ_thresh
    alpha = alpha.float()
    return alpha


class CVO_sampler_lmdb:
    """Data sampling"""

    all_keys = ["imgs", "imgs_blur", "fflows", "bflows", "delta_fflows", "delta_bflows"]

    def __init__(self, data_root, keys=None, split=None):
        if split == "extended":
            self.db_path = osp.join(data_root, "cvo_test_extended.lmdb")
        elif split == "train":
            self.db_path = osp.join(data_root, "cvo_train.lmdb")
        else:
            self.db_path = osp.join(data_root, "cvo_test.lmdb")
        self.split = split

        # breakpoint()
        # self.deserialize_func = pa.deserialize if "train" in self.split else pkl.loads
        self.deserialize_func = pkl.loads

        self.env = lmdb.open(
            self.db_path,
            subdir=os.path.isdir(self.db_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.env.begin(write=False) as txn:
            self.samples = self.deserialize_func(txn.get(b"__samples__"))
            self.length = len(self.samples)

        print(self.length)
        self.keys = self.all_keys if keys is None else [x.lower() for x in keys]
        self._check_keys(self.keys)

    def _check_keys(self, keys):
        # check keys are supported:
        for k in keys:
            assert k in self.all_keys, f"Invalid key value: {k}"

    def __len__(self):
        return self.length

    def sample(self, index):
        sample = OrderedDict()
        with self.env.begin(write=False) as txn:
            for k in self.keys:
                key = "{:05d}_{:s}".format(index, k)
                value = self.deserialize_func(txn.get(key.encode()))
                if "flow" in key and self.split in ["clean", "final"]:  # Convert Int to Floating
                    value = value.astype(np.float32)
                    value = (value - 2**15) / 128.0
                if "imgs" in k:
                    k = "imgs"
                sample[k] = value
        return sample


class CVO(Dataset):
    all_keys = ["fflows", "bflows"]

    def __init__(self, data_root, keys=None, split="clean", crop_size=256, debug=False):
        keys = self.all_keys if keys is None else [x.lower() for x in keys]
        self._check_keys(keys)
        if split == "final":
            keys.append("imgs_blur")
        else:
            keys.append("imgs")
        self.split = split

        self.data_root = data_root
        self.sampler = CVO_sampler_lmdb(data_root, keys, split)

        self.debug = debug

        print(f"Found {self.sampler.length} samples for CVO {split}")

    def __getitem__(self, index):
        sample = self.sampler.sample(index)

        try:
            if self.split in ["clean", "final"]:
                depth_path = os.path.join(self.data_root, "cvo_test_depth", f"{index:05d}.npy")
                videodepth = np.load(depth_path)
                videodepth = torch.from_numpy(videodepth).float()
                videodepth = rearrange(videodepth, "t h w -> t () h w")
            elif self.split == "extended":
                depth_path = os.path.join(self.data_root, "cvo_test_extended_depth", f"{index:05d}.npy")
                videodepth = np.load(depth_path)
                videodepth = torch.from_numpy(videodepth).float()
                videodepth = rearrange(videodepth, "t h w -> t () h w")
            else:
                videodepth = None
        except:
            videodepth = None
            
        video = torch.from_numpy(sample["imgs"].copy())
        video = rearrange(video, "h w (t c) -> t c h w", c=3)

        # NOTE concat and flip video
        video = torch.flip(video, dims=[0])  # flip temporal=
        if videodepth is not None:
            videodepth = torch.flip(videodepth, dims=[0])  # flip temporal

        fflow = torch.from_numpy(sample["fflows"].copy())
        fflow = rearrange(fflow, "h w (t c) -> t h w c", c=2)[-1]  # 0->6

        bflow = torch.from_numpy(sample["bflows"].copy())
        bflow = rearrange(bflow, "h w (t c) -> t h w c", c=2)[-1]  # 6->0

        # breakpoint()

        if self.split in ["clean", "final"]:
            thresh_1 = 0.01
            thresh_2 = 0.5
        elif self.split == "extended":
            thresh_1 = 0.1
            thresh_2 = 0.5
        else:
            raise ValueError(f"Unknown split {self.split}")

        alpha = get_alpha_consistency(bflow[None], fflow[None], thresh_1=thresh_1, thresh_2=thresh_2)[0]

        T, _, H, W = video.shape

        segs = torch.ones(T, 1, H, W).float()

        trajectory = torch.zeros(T, 1, 2).float()
        visibility = torch.zeros(T, 1).float()

        data = DeltaData(
            video=video,
            videodepth=videodepth,
            segmentation=segs,
            trajectory=trajectory,
            visibility=visibility,
            flow=bflow,
            flow_alpha=alpha,
            seq_name=f"{index:05d}",
        )

        return data

    def _check_keys(self, keys):
        # check keys are supported:
        for k in keys:
            assert k in self.all_keys, f"Invalid key value: {k}"

    def __len__(self):
        if self.debug:
            return 100

        return len(self.sampler)


def create_optical_flow_dataset(args):
    dataset = CVO(args.data_root, split=args.split)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    return dataloader
