# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import io
import os
import pickle
from typing import Mapping, Tuple, Union

import mediapy as media
import numpy as np
import torch
from densetrack3d.datasets.utils import DeltaData
from PIL import Image


DatasetElement = Mapping[str, Mapping[str, Union[np.ndarray, str]]]


def resize_video(video: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
    """Resize a video to output_size."""
    # If you have a GPU, consider replacing this with a GPU-enabled resize op,
    # such as a jitted jax.image.resize.  It will make things faster.
    return media.resize_video(video, output_size)


def sample_queries_first(
    target_occluded: np.ndarray,
    target_points: np.ndarray,
    frames: np.ndarray,
) -> Mapping[str, np.ndarray]:
    """Package a set of frames and tracks for use in TAPNet evaluations.
    Given a set of frames and tracks with no query points, use the first
    visible point in each track as the query.
    Args:
      target_occluded: Boolean occlusion flag, of shape [n_tracks, n_frames],
        where True indicates occluded.
      target_points: Position, of shape [n_tracks, n_frames, 2], where each point
        is [x,y] scaled between 0 and 1.
      frames: Video tensor, of shape [n_frames, height, width, 3].  Scaled between
        -1 and 1.
    Returns:
      A dict with the keys:
        video: Video tensor of shape [1, n_frames, height, width, 3]
        query_points: Query points of shape [1, n_queries, 3] where
          each point is [t, y, x] scaled to the range [-1, 1]
        target_points: Target points of shape [1, n_queries, n_frames, 2] where
          each point is [x, y] scaled to the range [-1, 1]
    """
    valid = np.sum(~target_occluded, axis=1) > 0
    target_points = target_points[valid, :]
    target_occluded = target_occluded[valid, :]

    query_points = []
    for i in range(target_points.shape[0]):
        index = np.where(target_occluded[i] == 0)[0][0]
        x, y = target_points[i, index, 0], target_points[i, index, 1]
        query_points.append(np.array([index, y, x]))  # [t, y, x]
    query_points = np.stack(query_points, axis=0)

    return {
        "video": frames[np.newaxis, ...],
        "query_points": query_points[np.newaxis, ...],
        "target_points": target_points[np.newaxis, ...],
        "occluded": target_occluded[np.newaxis, ...],
    }


def sample_queries_strided(
    target_occluded: np.ndarray,
    target_points: np.ndarray,
    frames: np.ndarray,
    query_stride: int = 5,
) -> Mapping[str, np.ndarray]:
    """Package a set of frames and tracks for use in TAPNet evaluations.

    Given a set of frames and tracks with no query points, sample queries
    strided every query_stride frames, ignoring points that are not visible
    at the selected frames.

    Args:
      target_occluded: Boolean occlusion flag, of shape [n_tracks, n_frames],
        where True indicates occluded.
      target_points: Position, of shape [n_tracks, n_frames, 2], where each point
        is [x,y] scaled between 0 and 1.
      frames: Video tensor, of shape [n_frames, height, width, 3].  Scaled between
        -1 and 1.
      query_stride: When sampling query points, search for un-occluded points
        every query_stride frames and convert each one into a query.

    Returns:
      A dict with the keys:
        video: Video tensor of shape [1, n_frames, height, width, 3].  The video
          has floats scaled to the range [-1, 1].
        query_points: Query points of shape [1, n_queries, 3] where
          each point is [t, y, x] scaled to the range [-1, 1].
        target_points: Target points of shape [1, n_queries, n_frames, 2] where
          each point is [x, y] scaled to the range [-1, 1].
        trackgroup: Index of the original track that each query point was
          sampled from.  This is useful for visualization.
    """
    tracks = []
    occs = []
    queries = []
    trackgroups = []
    total = 0
    trackgroup = np.arange(target_occluded.shape[0])
    for i in range(0, target_occluded.shape[1], query_stride):
        mask = target_occluded[:, i] == 0
        query = np.stack(
            [
                i * np.ones(target_occluded.shape[0:1]),
                target_points[:, i, 1],
                target_points[:, i, 0],
            ],
            axis=-1,
        )
        queries.append(query[mask])
        tracks.append(target_points[mask])
        occs.append(target_occluded[mask])
        trackgroups.append(trackgroup[mask])
        total += np.array(np.sum(target_occluded[:, i] == 0))

    return {
        "video": frames[np.newaxis, ...],
        "query_points": np.concatenate(queries, axis=0)[np.newaxis, ...],
        "target_points": np.concatenate(tracks, axis=0)[np.newaxis, ...],
        "occluded": np.concatenate(occs, axis=0)[np.newaxis, ...],
        "trackgroup": np.concatenate(trackgroups, axis=0)[np.newaxis, ...],
    }


class TapVidDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root,
        dataset_type="davis",
        resize_to_256=True,
        resize_to_cotracker=False,
        queried_first=True,
    ):
        self.dataset_type = dataset_type
        self.resize_to_256 = resize_to_256
        self.resize_to_cotracker = resize_to_cotracker
        self.queried_first = queried_first
        if self.dataset_type == "kinetics":
            all_paths = glob.glob(os.path.join(data_root, "*_of_0010.pkl"))
            points_dataset = []
            for pickle_path in all_paths:
                with open(pickle_path, "rb") as f:
                    data = pickle.load(f)
                    points_dataset = points_dataset + data
            self.points_dataset = points_dataset
        else:
            with open(data_root, "rb") as f:
                self.points_dataset = pickle.load(f)
            if self.dataset_type == "davis":
                self.video_names = list(self.points_dataset.keys())

        # NOTE load external depth
        # breakpoint()
        if os.path.exists(data_root.replace("tapvid_davis.pkl", "depth_tapvid_davis.pkl")):
            with open(data_root.replace("tapvid_davis.pkl", "depth_tapvid_davis.pkl"), "rb") as f:
                depth_data = pickle.load(f)

            for k, v in depth_data.items():
                if k in self.points_dataset.keys():
                    self.points_dataset[k]["depth"] = v

        print("found %d unique videos in %s" % (len(self.points_dataset), data_root))

    def __getitem__(self, index):
        if self.dataset_type == "davis":
            video_name = self.video_names[index]
        else:
            video_name = index
        video = self.points_dataset[video_name].copy()
        frames = video["video"]

        if isinstance(frames[0], bytes):
            # TAP-Vid is stored and JPEG bytes rather than `np.ndarray`s.
            def decode(frame):
                byteio = io.BytesIO(frame)
                img = Image.open(byteio)
                return np.array(img)

            frames = np.array([decode(frame) for frame in frames])

        target_points = self.points_dataset[video_name]["points"].copy()

        if self.resize_to_cotracker:
            frames = resize_video(frames, [384, 512])
            target_points *= np.array([512, 384])
        elif self.resize_to_256:
            frames = resize_video(frames, [256, 256])
            target_points *= np.array([256, 256])
        else:
            target_points *= np.array([frames.shape[2], frames.shape[1]])

        T, H, W, C = frames.shape
        N, T, D = target_points.shape

        target_occ = self.points_dataset[video_name]["occluded"].copy()
        if self.queried_first:
            converted = sample_queries_first(target_occ, target_points, frames)
        else:
            converted = sample_queries_strided(target_occ, target_points, frames)
        assert converted["target_points"].shape[1] == converted["query_points"].shape[1]

        trajs = torch.from_numpy(converted["target_points"])[0].permute(1, 0, 2).float()  # T, N, D

        rgbs = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
        segs = torch.ones(T, 1, H, W).float()
        visibles = torch.logical_not(torch.from_numpy(converted["occluded"]))[0].permute(1, 0)  # T, N
        query_points = torch.from_numpy(converted["query_points"])[0]  # T, N

        if "depth" in video.keys():
            depths = torch.from_numpy(video["depth"])[..., None].permute(0, 3, 1, 2).float()
        else:
            depths = None

            # # NOTE debug
            # rgbs = rgbs[:24]
            # depths = depths[:24]
            # segs = segs[:24]
            # mask_valid = query_points[:, 0] < 24
            # query_points = query_points[mask_valid]
            # trajs = trajs[:24, mask_valid]
            # visibles = visibles[:24, mask_valid]

        data = DeltaData(
            video=rgbs,
            videodepth=depths,
            segmentation=segs,
            trajectory=trajs,
            visibility=visibles,
            seq_name=str(video_name),
            query_points=query_points,
            dataset_name="tapvid_davis",
        )

        # print("debug", data.dataset_name, data.seq_name)
        return data

    def __len__(self):
        return len(self.points_dataset)
