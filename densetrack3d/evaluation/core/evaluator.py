# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from collections import defaultdict
from typing import Optional

# from collections import namedtuple
import cv2
import mediapy as media
import numpy as np
import torch
import torch.nn.functional as F
from densetrack3d.datasets.utils import DeltaData, dataclass_to_cuda_
from densetrack3d.evaluation.core.cvo_metrics import compute_metrics as compute_cvo_metrics
from densetrack3d.evaluation.core.eval_utils import compute_lsfodyssey_metrics, compute_tapvid_metrics
from densetrack3d.evaluation.core.sintel_metrics import compute_sintel_metrics
from densetrack3d.evaluation.core.tapvid3d_metrics import compute_tapvid3d_metrics
from densetrack3d.models.geometry_utils import project_3d2d, reproject_2d3d
from densetrack3d.models.model_utils import (
    bilinear_sample2d,
    bilinear_sampler,
    dense_to_sparse_tracks_3d_in_3dspace,
    get_grid,
    get_points_on_a_grid,
    reduce_masked_mean,
    smart_cat,
)
from densetrack3d.utils.io import create_folder, write_frame_np, write_video
from densetrack3d.utils.visualizer import Visualizer, flow_to_rgb
from einops import rearrange
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class Evaluator:
    """
    A class defining the CoTracker evaluator.
    """

    def __init__(self, exp_dir) -> None:
        # Visualization
        self.exp_dir = exp_dir

    def compute_metrics(self, metrics, sample, pred_traj_2d, pred_visibility, dataset_name, resolution=(384, 512)):

        if "tapvid" in dataset_name:
            B, T, N, D = sample.trajectory.shape
            traj = sample.trajectory.clone()
            thr = 0.9

            if pred_visibility is None:
                logging.warning("visibility is NONE")
                pred_visibility = torch.zeros_like(sample.visibility)

            if not pred_visibility.dtype == torch.bool:
                pred_visibility = pred_visibility > thr

            query_points = sample.query_points.clone().cpu().numpy()

            pred_visibility = pred_visibility[:, :, :N]
            pred_traj_2d = pred_traj_2d[:, :, :N]

            gt_tracks = traj.permute(0, 2, 1, 3).cpu().numpy()
            gt_occluded = torch.logical_not(sample.visibility.clone().permute(0, 2, 1)).cpu().numpy()

            pred_occluded = torch.logical_not(pred_visibility.clone().permute(0, 2, 1)).cpu().numpy()
            pred_tracks = pred_traj_2d.permute(0, 2, 1, 3).cpu().numpy()

            if resolution[0] != 256:
                pred_tracks[..., 1] *= 256 / resolution[0]
                gt_tracks[..., 1] *= 256 / resolution[0]
            if resolution[1] != 256:
                pred_tracks[..., 0] *= 256 / resolution[1]
                gt_tracks[..., 0] *= 256 / resolution[1]

            out_metrics = compute_tapvid_metrics(
                query_points,
                gt_occluded,
                gt_tracks,
                pred_occluded,
                pred_tracks,
                query_mode="strided" if "strided" in dataset_name else "first",
            )

            metrics[sample.seq_name[0]] = out_metrics
            for metric_name in out_metrics.keys():
                if "avg" not in metrics:
                    metrics["avg"] = {}
                metrics["avg"][metric_name] = np.mean([v[metric_name] for k, v in metrics.items() if k != "avg"])

        elif dataset_name == "dynamic_replica" or dataset_name == "pointodyssey":
            *_, N, _ = sample.trajectory.shape
            B, T, N = sample.visibility.shape
            H, W = sample.video.shape[-2:]
            device = sample.video.device

            out_metrics = {}

            d_vis_sum = d_occ_sum = d_sum_all = 0.0
            thrs = [1, 2, 4, 8, 16]
            sx_ = (W - 1) / 255.0
            sy_ = (H - 1) / 255.0
            sc_py = np.array([sx_, sy_]).reshape([1, 1, 2])
            sc_pt = torch.from_numpy(sc_py).float().to(device)
            __, first_visible_inds = torch.max(sample.visibility, dim=1)

            frame_ids_tensor = torch.arange(T, device=device)[None, :, None].repeat(B, 1, N)
            start_tracking_mask = frame_ids_tensor > (first_visible_inds.unsqueeze(1))

            for thr in thrs:
                d_ = (
                    torch.norm(
                        pred_traj_2d[..., :2] / sc_pt - sample.trajectory[..., :2] / sc_pt,
                        dim=-1,
                    )
                    < thr
                ).float()  # B,S-1,N
                d_occ = reduce_masked_mean(d_, (1 - sample.visibility) * start_tracking_mask).item() * 100.0
                d_occ_sum += d_occ
                out_metrics[f"accuracy_occ_{thr}"] = d_occ

                d_vis = reduce_masked_mean(d_, sample.visibility * start_tracking_mask).item() * 100.0
                d_vis_sum += d_vis
                out_metrics[f"accuracy_vis_{thr}"] = d_vis

                d_all = reduce_masked_mean(d_, start_tracking_mask).item() * 100.0
                d_sum_all += d_all
                out_metrics[f"accuracy_{thr}"] = d_all

            d_occ_avg = d_occ_sum / len(thrs)
            d_vis_avg = d_vis_sum / len(thrs)
            d_all_avg = d_sum_all / len(thrs)

            sur_thr = 50
            dists = torch.norm(
                pred_traj_2d[..., :2] / sc_pt - sample.trajectory[..., :2] / sc_pt,
                dim=-1,
            )  # B,S,N
            dist_ok = 1 - (dists > sur_thr).float() * sample.visibility  # B,S,N
            survival = torch.cumprod(dist_ok, dim=1)  # B,S,N
            out_metrics["survival"] = torch.mean(survival).item() * 100.0

            out_metrics["accuracy_occ"] = d_occ_avg
            out_metrics["accuracy_vis"] = d_vis_avg
            out_metrics["accuracy"] = d_all_avg

            metrics[sample.seq_name[0]] = out_metrics
            for metric_name in out_metrics.keys():
                if "avg" not in metrics:
                    metrics["avg"] = {}
                metrics["avg"][metric_name] = float(
                    np.mean([v[metric_name] for k, v in metrics.items() if k != "avg"])
                )

            # logging.info(f"Metrics: {out_metrics}")
            # logging.info(f"avg: {metrics['avg']}")
            # print("metrics", out_metrics)
            # print("avg", metrics["avg"])
        # return out_metrics

    def compute_metrics_3d(self, metrics, sample, pred_traj_3d, pred_visibility, dataset_name):

        if pred_visibility.dtype != torch.bool:
            pred_visibility = pred_visibility > 0.9  # FIXME tune visib here
            # pred_visibility = torch.ones_like(pred_visibility).bool()

        if "lsfodyssey" in dataset_name:
            trajs_g = sample.trajectory3d
            valids = sample.valid
            vis_g = sample.visibility
            intrs = sample.intrs

            gt_traj_3d = reproject_2d3d(trajs_g, intrs)

            # breakpoint()

            intrinsics_params = torch.stack(
                [intrs[0, 0, 0, 0], intrs[0, 0, 1, 1], intrs[0, 0, 0, 2], intrs[0, 0, 1, 2]], dim=-1
            )
            # breakpoint()
            query_points = sample.query_points.cpu().numpy()
            out_metrics, scaled_pred_traj_3d = compute_tapvid3d_metrics(
                gt_occluded=np.logical_not(vis_g.cpu().numpy()),
                gt_tracks=gt_traj_3d.cpu().numpy(),
                pred_occluded=np.logical_not(pred_visibility.cpu().numpy()),
                pred_tracks=pred_traj_3d.cpu().numpy(),
                intrinsics_params=intrinsics_params.cpu().numpy(),
                scaling="median",  # 'per_trajectory', #
                query_points=query_points,
                order="b t n",
                use_fixed_metric_threshold=False,
                return_scaled_pred=True,
            )

            # print("metric", out_metrics)

        elif "tapvid3d" in dataset_name:
            trajs_g = sample.trajectory3d
            valids = sample.valid
            vis_g = sample.visibility
            intrs = sample.intrs

            # print(N, pred_trajectory.shape, intrs.shape)
            # pred_traj_3d = reproject_2d3d(pred_trajectory, intrs)
            intrinsics_params = torch.stack(
                [intrs[0, 0, 0, 0], intrs[0, 0, 1, 1], intrs[0, 0, 0, 2], intrs[0, 0, 1, 2]], dim=-1
            )

            query_points = sample.query_points.cpu().numpy()
            out_metrics, scaled_pred_traj_3d = compute_tapvid3d_metrics(
                gt_occluded=np.logical_not(vis_g.cpu().numpy()),
                gt_tracks=trajs_g.cpu().numpy(),
                pred_occluded=np.logical_not(pred_visibility.cpu().numpy()),
                pred_tracks=pred_traj_3d.cpu().numpy(),
                intrinsics_params=intrinsics_params.cpu().numpy(),
                scaling="median",  # 'per_trajectory', #
                query_points=query_points,
                order="b t n",
                use_fixed_metric_threshold=False,
                return_scaled_pred=True,
            )

            # print("metric", out_metrics)

        elif "kubric3d" in dataset_name:
            intrs = sample.intrs

            dense_trajs = sample.dense_trajectory
            dense_trajs_d = sample.dense_trajectory_d
            dense_trajs_uvd = torch.cat([dense_trajs, dense_trajs_d], dim=2)
            dense_trajs_uvd = rearrange(dense_trajs_uvd, "b t c h w -> b t (h w) c")
            gt_traj_3d = reproject_2d3d(dense_trajs_uvd, intrs)

            dense_trajs_vis = sample.flow_alpha
            dense_trajs_vis = rearrange(dense_trajs_vis, "b t h w -> b t (h w)")

            intrinsics_params = torch.stack(
                [intrs[0, 0, 0, 0], intrs[0, 0, 1, 1], intrs[0, 0, 0, 2], intrs[0, 0, 1, 2]], dim=-1
            )

            query_points = (
                torch.cat([torch.zeros_like(dense_trajs_uvd[:, 0, :, 0:1]), dense_trajs_uvd[:, 0, :, :3]], dim=-1)
                .cpu()
                .numpy()
            )
            out_metrics, scaled_pred_traj_3d = compute_tapvid3d_metrics(
                gt_occluded=np.logical_not(dense_trajs_vis.cpu().numpy()),
                gt_tracks=gt_traj_3d.cpu().numpy(),
                pred_occluded=np.logical_not(pred_visibility.cpu().numpy()),
                pred_tracks=pred_traj_3d.cpu().numpy(),
                intrinsics_params=intrinsics_params.cpu().numpy(),
                scaling="median",  # 'per_trajectory', #
                query_points=query_points,
                order="b t n",
                use_fixed_metric_threshold=False,
                return_scaled_pred=True,
            )

        metrics[sample.seq_name[0]] = out_metrics
        for metric_name in out_metrics.keys():
            if "avg" not in metrics:
                metrics["avg"] = {}
            metrics["avg"][metric_name] = np.mean([v[metric_name] for k, v in metrics.items() if k != "avg"])

        return scaled_pred_traj_3d

    @torch.no_grad()
    def evaluate_sequence(
        self,
        model,
        test_dataloader: torch.utils.data.DataLoader,
        dataset_name: str,
        train_mode=False,
        visualize_every: int = -1,
        writer: Optional[SummaryWriter] = None,
        step: Optional[int] = 0,
        is_sparse: bool = True,
        verbose: bool = False,
        use_2d_only: bool = False,
    ):

        if visualize_every > 0:
            vis = Visualizer(
                save_dir=self.exp_dir,
                fps=10,
                show_first_frame=0,
            )

        metrics = {}

        for ind, sample in enumerate(tqdm(test_dataloader)):
            if isinstance(sample, tuple):
                sample, gotit = sample
                if not all(gotit):
                    print("batch is None")
                    continue

            dataclass_to_cuda_(sample)

            if "tapvid" in dataset_name:
                queries = sample.query_points.clone().float()

                queries = torch.stack(
                    [
                        queries[:, :, 0],
                        queries[:, :, 2],
                        queries[:, :, 1],
                    ],
                    dim=2,
                )
            else:
                queries = torch.cat(
                    [
                        torch.zeros_like(sample.trajectory[:, 0, :, :1]),
                        sample.trajectory[:, 0],
                    ],
                    dim=2,
                )

            # get the 3D queries
            depth_interp = []
            for i in range(queries.shape[1]):
                depth_interp_i = bilinear_sample2d(
                    sample.videodepth[0, queries[:, i : i + 1, 0].long()],
                    queries[:, i : i + 1, 1],
                    queries[:, i : i + 1, 2],
                )
                depth_interp.append(depth_interp_i)

            depth_interp = torch.cat(depth_interp, dim=1)
            queries = smart_cat(queries, depth_interp, dim=-1)

            resolution = (sample.video.shape[-2], sample.video.shape[-1])

            traj_e, traj_d_e, vis_e = model(
                video=sample.video,
                videodepth=sample.videodepth,
                queries=queries,
                depth_init=sample.videodepth[:, 0],
                intrs=sample.intrs,
                return_3d=True,
                is_sparse=is_sparse,
                use_2d_only=use_2d_only,
            )

            self.compute_metrics(metrics, sample, traj_e, vis_e, dataset_name, resolution=resolution)

            if verbose:
                logging.info(f"Metrics: {metrics[sample.seq_name[0]]}")
                logging.info(f"avg: {metrics['avg']}")
                print("metrics", metrics[sample.seq_name[0]])
                print("avg", metrics["avg"])

            if visualize_every > 0 and ind % visualize_every == 0:
                vis.visualize(
                    sample.video,
                    traj_e[..., :2],
                    vis_e.unsqueeze(-1),
                    filename=dataset_name + "_" + sample.seq_name[0],
                    writer=writer,
                    step=step,
                )
        return metrics

    @torch.no_grad()
    def evaluate_sequence_2d(
        self,
        model,
        test_dataloader: torch.utils.data.DataLoader,
        dataset_name: str,
        train_mode=False,
        visualize_every: int = 1,
        writer: Optional[SummaryWriter] = None,
        step: Optional[int] = 0,
        is_sparse: bool = True,
    ):
        metrics = {}

        vis = Visualizer(
            save_dir=self.exp_dir,
            fps=10,
        )

        for ind, sample in enumerate(tqdm(test_dataloader)):
            if isinstance(sample, tuple):
                sample, gotit = sample
                if not all(gotit):
                    print("batch is None")
                    continue
            if torch.cuda.is_available():
                dataclass_to_cuda_(sample)
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

            if (
                not train_mode
                and hasattr(model, "sequence_len")
                and (sample.visibility[:, : model.sequence_len].sum() == 0)
            ):
                print(f"skipping batch {ind}")
                continue

            if "tapvid" in dataset_name:
                queries = sample.query_points.clone().float()

                queries = torch.stack(
                    [
                        queries[:, :, 0],
                        queries[:, :, 2],
                        queries[:, :, 1],
                    ],
                    dim=2,
                ).to(device)
            else:
                queries = torch.cat(
                    [
                        torch.zeros_like(sample.trajectory[:, 0, :, :1]),
                        sample.trajectory[:, 0],
                    ],
                    dim=2,
                ).to(device)

            resolution = (sample.video.shape[-2], sample.video.shape[-1])
            pred_tracks = model(video=sample.video, queries=queries, is_sparse=is_sparse)

            # pred_tracks = model(sample.video, queries)

            # breakpoint()
            if "strided" in dataset_name:
                inv_video = sample.video.flip(1).clone()
                inv_queries = queries.clone()
                inv_queries[:, :, 0] = inv_video.shape[1] - inv_queries[:, :, 0] - 1

                pred_trj, pred_vsb = pred_tracks
                inv_pred_trj, inv_pred_vsb = model(inv_video, inv_queries)

                inv_pred_trj = inv_pred_trj.flip(1)
                inv_pred_vsb = inv_pred_vsb.flip(1)

                mask = pred_trj == 0

                pred_trj[mask] = inv_pred_trj[mask]
                pred_vsb[mask[:, :, :, 0]] = inv_pred_vsb[mask[:, :, :, 0]]

                pred_tracks = pred_trj, pred_vsb

            if dataset_name == "badja" or dataset_name == "fastcapture":
                seq_name = sample.seq_name[0]
            else:
                seq_name = str(ind)
            if ind % visualize_every == 0 and visualize_every > 0:
                vis.visualize(
                    sample.video,
                    pred_tracks[0][..., :2],
                    pred_tracks[1].unsqueeze(-1),
                    filename=dataset_name + "_" + seq_name,
                    writer=writer,
                    step=step,
                )

            pred_tracks = tuple([pred_tracks[0][..., :2], pred_tracks[1]])
            self.compute_metrics(metrics, sample, pred_tracks, dataset_name, resolution=resolution)
        return metrics

    @torch.no_grad()
    def evaluate_sequence_2d_consistency(
        self,
        model,
        test_dataloader: torch.utils.data.DataLoader,
        dataset_name: str,
        train_mode=False,
        visualize_every: int = 1,
        writer: Optional[SummaryWriter] = None,
        step: Optional[int] = 0,
        is_sparse: bool = True,
    ):
        metrics = {}

        vis = Visualizer(save_dir=self.exp_dir, fps=7, show_first_frame=False, linewidth=1)

        grid_xy = get_points_on_a_grid((36, 48), (384, 512)).long().float()
        queries = torch.cat([torch.zeros_like(grid_xy[:, :, :1]), grid_xy], dim=2).cuda()  # B, N, C

        for ind, sample in enumerate(tqdm(test_dataloader)):
            if isinstance(sample, tuple):
                sample, gotit = sample
                if not all(gotit):
                    print("batch is None")
                    continue
            if torch.cuda.is_available():
                dataclass_to_cuda_(sample)
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

            video = sample.video
            video_chunks = torch.split(video, 16, dim=1)

            for chunk_id, video_sample in enumerate(video_chunks):


                n_queries = queries.shape[1]

                resolution = video_sample.shape[-2:]
                pred_trj, pred_vsb = model(video=video_sample, queries=queries, is_sparse=is_sparse)

                pred_trj = pred_trj[:, :, :n_queries]
                pred_vsb = pred_vsb[:, :, :n_queries]

                last_trj, last_vsb = pred_trj[0, -1], pred_vsb[0, -1]  # N 2

                if last_vsb.dtype != torch.bool:
                    last_vsb = last_vsb > 0.9
                valid_inv = (
                    last_vsb
                    & (last_trj[:, 0] >= 0)
                    & (last_trj[:, 1] >= 0)
                    & (last_trj[:, 0] < resolution[1])
                    & (last_trj[:, 1] < resolution[0])
                )

                if valid_inv.sum() == 0:
                    continue

                last_trj = last_trj[valid_inv]

                inv_queries = torch.cat([torch.zeros_like(last_trj[..., 0:1]), last_trj], dim=-1)[None]  # B N 3

                n_inv_queries = inv_queries.shape[1]
                inv_video = video_sample.flip(1).clone()
                inv_pred_trj, inv_pred_vsb = model(inv_video, inv_queries, is_sparse=is_sparse)

                inv_pred_trj = inv_pred_trj[:, :, :n_inv_queries].flip(1)
                inv_pred_vsb = inv_pred_vsb[:, :, :n_inv_queries].flip(1)

                pred_trj_ = pred_trj[:, :, valid_inv]
                pred_vsb_ = pred_vsb[:, :, valid_inv]

                fw_video1 = vis.visualize(
                    video_sample,
                    pred_trj,
                    pred_vsb.unsqueeze(-1),
                    filename=f"test_{ind}_fw",
                    default_color=np.array([0, 255, 0]),
                    save_video=False,
                )

                fw_video = vis.visualize(
                    video_sample,
                    pred_trj_,
                    pred_vsb_.unsqueeze(-1),
                    filename=f"test_{ind}_fw",
                    default_color=np.array([0, 255, 0]),
                    save_video=False,
                )

                bw_video = vis.visualize(
                    fw_video,
                    inv_pred_trj,
                    inv_pred_vsb.unsqueeze(-1),
                    filename=f"test_{ind}_fw_bw",
                    default_color=np.array([255, 0, 0]),
                    save_video=False,
                )
                video2d_viz = bw_video[0].permute(0, 2, 3, 1).cpu().numpy()
                media.write_video(
                    f"results/test_cycle/{sample.seq_name[0]}_{chunk_id}.mp4",
                    media.resize_video(video2d_viz, (512, 512)),
                    fps=12,
                )
                video2d_viz2 = fw_video1[0].permute(0, 2, 3, 1).cpu().numpy()
                media.write_video(
                    f"results/test_cycle/{sample.seq_name[0]}_{chunk_id}_fw.mp4",
                    media.resize_video(video2d_viz2, (512, 512)),
                    fps=12,
                )

                pseudo_sample_gt = DeltaData(
                    video=None,
                    trajectory=pred_trj_,
                    visibility=pred_vsb_,
                    query_points=queries[:, valid_inv],
                    seq_name=[sample.seq_name[0] + f"_{str(chunk_id)}"],
                )
                self.compute_metrics(
                    metrics, pseudo_sample_gt, (inv_pred_trj, inv_pred_vsb), dataset_name, resolution=resolution
                )
        return metrics

    @torch.no_grad()
    def evaluate_sequence_3d(
        self,
        model,
        test_dataloader: torch.utils.data.DataLoader,
        dataset_name: str,
        train_mode=False,
        writer: Optional[SummaryWriter] = None,
        step: Optional[int] = 0,
        visualize_every: Optional[int] = 1,
        is_vis: bool = True,
        is_sparse: bool = True,
        lift_3d: bool = False,  # sample depth from depth map
        verbose: bool = False,
    ):
        metrics = {}

        if is_vis:
            vis = Visualizer(
                save_dir=self.exp_dir,
                fps=10,
                show_first_frame=0,
            )

        for ind, sample in enumerate(test_dataloader):
            if isinstance(sample, tuple):
                sample, gotit = sample
                if not all(gotit):
                    print("batch is None")
                    continue

            dataclass_to_cuda_(sample)

            if "lsfodyssey" in dataset_name or "tapvid3d" in dataset_name:
                queries = sample.query_points.clone().float()  # B N 3
            else:
                # NOTE 3D queries
                queries = torch.cat(
                    [
                        torch.zeros_like(sample.trajectory3d[:, 0, :, :1]),
                        sample.trajectory3d[:, 0],
                    ],
                    dim=2,
                )

            n_queries = queries.shape[1]

            intrs = sample.intrs

            traj_e, traj_d_e, vis_e = model(
                video=sample.video,
                videodepth=sample.videodepth,
                queries=queries,
                depth_init=sample.videodepth[:, 0],
                intrs=sample.intrs,
                return_3d=True,
                is_sparse=is_sparse,
                lift_3d=lift_3d,
            )

            traj_e = traj_e[:, :, :n_queries]
            traj_d_e = traj_d_e[:, :, :n_queries]
            vis_e = vis_e[:, :, :n_queries]

            if "tapvid3d" in dataset_name:
                # NOTE tracking backward
                inv_video = sample.video.flip(1).clone()
                inv_videodepth = sample.videodepth.flip(1).clone()
                inv_queries = queries.clone()
                inv_queries[:, :, 0] = inv_video.shape[1] - inv_queries[:, :, 0] - 1

                inv_traj_e, inv_traj_d_e, inv_vis_e = model(
                    video=inv_video,
                    videodepth=inv_videodepth,
                    queries=inv_queries,
                    depth_init=inv_videodepth[:, 0],
                    return_3d=True,
                    is_sparse=is_sparse,
                    lift_3d=lift_3d,
                )

                inv_traj_e = inv_traj_e[:, :, :n_queries].flip(1)
                inv_traj_d_e = inv_traj_d_e[:, :, :n_queries].flip(1)
                inv_vis_e = inv_vis_e[:, :, :n_queries].flip(1)

                arange = torch.arange(sample.video.shape[1], device=queries.device)[None, :, None]
                mask = (arange < queries[:, None, :, 0]).unsqueeze(-1).repeat(1, 1, 1, inv_traj_e.shape[-1])

                traj_e[mask] = inv_traj_e[mask]
                traj_d_e[mask[:, :, :, 0]] = inv_traj_d_e[mask[:, :, :, 0]]
                vis_e[mask[:, :, :, 0]] = inv_vis_e[mask[:, :, :, 0]]

            traj_uvd = torch.cat([traj_e, traj_d_e], dim=-1)
            traj_3d = reproject_2d3d(traj_uvd, intrs)

            scaled_pred_traj_3d = self.compute_metrics_3d(metrics, sample, traj_3d, vis_e, dataset_name)
            if verbose:
                print("Avg:", metrics["avg"])
        return metrics

    @torch.no_grad()
    def evaluate_sequence_3d_using_cotracker2D(
        self,
        model,
        test_dataloader: torch.utils.data.DataLoader,
        dataset_name: str,
        train_mode=False,
        writer: Optional[SummaryWriter] = None,
        step: Optional[int] = 0,
        visualize_every: Optional[int] = 1,
        is_vis=True,
        is_sparse=True,
    ):
        metrics = {}

        for ind, sample in enumerate(test_dataloader):
            # if ind != 1 :
            #    continue
            if isinstance(sample, tuple):
                sample, gotit = sample
                if not all(gotit):
                    print("batch is None")
                    continue
            if torch.cuda.is_available():
                device = torch.device("cuda")
                dataclass_to_cuda_(sample)
            else:
                device = torch.device("cpu")

            if (
                not train_mode
                and hasattr(model, "sequence_len")
                and (sample.visibility[:, : model.sequence_len].sum() == 0)
            ):
                print(f"skipping batch {ind}")
                continue

            if "lsfodyssey" in dataset_name or "tapvid3d" in dataset_name:
                queries = sample.query_points.clone().float()  # B N 3
            else:
                # NOTE 3D queries
                queries = torch.cat(
                    [
                        torch.zeros_like(sample.trajectory3d[:, 0, :, :1]),
                        sample.trajectory3d[:, 0],
                    ],
                    dim=2,
                )

            n_queries = queries.shape[1]

            preds = model.forward_lift3D(
                video=sample.video,
                videodepth=sample.videodepth,
                queries=queries,
                depth_init=sample.videodepth[:, 0],
                return_3d=True,
                is_sparse=is_sparse,
            )

            if "tapvid3d" in dataset_name:
                inv_video = sample.video.flip(1).clone()
                inv_videodepth = sample.videodepth.flip(1).clone()
                inv_queries = queries.clone()
                inv_queries[:, :, 0] = inv_video.shape[1] - inv_queries[:, :, 0] - 1

                inv_preds = model.forward_lift3D(
                    video=inv_video,
                    videodepth=inv_videodepth,
                    queries=inv_queries,
                    depth_init=inv_videodepth[:, 0],
                    return_3d=True,
                    is_sparse=is_sparse,
                )
                inv_tracks, inv_visibilities = inv_preds
                inv_tracks = inv_tracks[:, :, :n_queries]
                inv_visibilities = inv_visibilities[:, :, :n_queries]

                inv_tracks = inv_tracks.flip(1)
                inv_visibilities = inv_visibilities.flip(1)
                arange = torch.arange(sample.video.shape[1], device=queries.device)[None, :, None]

                mask = (arange < queries[:, None, :, 0]).unsqueeze(-1).repeat(1, 1, 1, inv_tracks.shape[-1])

                pred_tracks, pred_visibilities = preds
                pred_tracks = pred_tracks[:, :, :n_queries]
                pred_visibilities = pred_visibilities[:, :, :n_queries]

                # breakpoint()e
                pred_tracks[mask] = inv_tracks[mask]
                pred_visibilities[mask[:, :, :, 0]] = inv_visibilities[mask[:, :, :, 0]]

                preds = (pred_tracks, pred_visibilities)

            scaled_pred_traj_3d = self.compute_metrics_3d(metrics, sample, preds, dataset_name)

            print(metrics["avg"])
        return metrics

    @torch.no_grad()
    def evaluate_flow(
        self,
        model,
        test_dataloader: torch.utils.data.DataLoader,
        split: str = "clean",
        verbose: bool = False,
        dataset_name: str = "cvo",
    ):
        metrics = {}

        # vis = Visualizer(
        #     save_dir=self.exp_dir,
        #     fps=7,
        # )

        filter_indices = [
            70,
            77,
            93,
            96,
            140,
            143,
            162,
            172,
            174,
            179,
            187,
            215,
            236,
            284,
            285,
            293,
            330,
            358,
            368,
            402,
            415,
            458,
            483,
            495,
            534,
        ]

        for ind, sample in enumerate(tqdm(test_dataloader)):

            if ind in filter_indices and split in ["clean", "final"] and dataset_name == "cvo":
                continue
            if isinstance(sample, tuple):
                sample, gotit = sample
                if not all(gotit):
                    print("batch is None")
                    continue

            dataclass_to_cuda_(sample)

            flow, flow_alpha = model.forward_flow2d(
                video=sample.video,
                videodepth=sample.videodepth,
                dst_frame=-1 if dataset_name == "cvo" else 1,
            )

            gt_flow = sample.flow
            gt_alpha = sample.flow_alpha

            if dataset_name == "cvo":
                out_metrics = compute_cvo_metrics(
                    gt={"flow": gt_flow, "alpha": gt_alpha}, pred={"flow": flow, "alpha": flow_alpha}
                )
            else:
                gt_valid = sample.flow_valid
                out_metrics = compute_sintel_metrics(flow, gt_flow, gt_valid)

            metrics[sample.seq_name[0]] = out_metrics
            for metric_name in out_metrics.keys():
                if "avg" not in metrics:
                    metrics["avg"] = {}
                metrics["avg"][metric_name] = float(
                    np.nanmean([v[metric_name] for k, v in metrics.items() if k != "avg"])
                )

            # print("Current", out_metrics)
            if verbose:
                print("Current:", out_metrics)
                print("Avg:", metrics["avg"])
        # print("Final:", metrics["avg"])

        return metrics

    @torch.no_grad()
    def evaluate_flow3d(
        self,
        model,
        test_dataloader: torch.utils.data.DataLoader,
        train_mode=False,
        visualize_every: int = 1,
        writer: Optional[SummaryWriter] = None,
        step: Optional[int] = 0,
    ):
        metrics = {}

        for ind, sample in enumerate(tqdm(test_dataloader)):
            if isinstance(sample, tuple):
                sample, gotit = sample
                if not all(gotit):
                    print("batch is None")
                    continue

            dataclass_to_cuda_(sample)

            if (
                not train_mode
                and hasattr(model, "sequence_len")
                and (sample.visibility[:, : model.sequence_len].sum() == 0)
            ):
                print(f"skipping batch {ind}")
                continue

            pred_trajs_uvd, pred_vis = model.forward_flow3d(
                video=sample.video,
                videodepth=sample.videodepth,
            )

            pred_trajs_uvd = rearrange(pred_trajs_uvd, "b t c h w -> b t (h w) c")
            pred_vis = rearrange(pred_vis, "b t h w -> b t (h w)")

            scaled_pred_traj_3d = self.compute_metrics_3d(metrics, sample, pred_trajs_uvd, pred_vis, "kubric3d")

        print("avg", metrics["avg"])

        return metrics

    @torch.no_grad()
    def evaluate_flow_2d(
        self,
        model,
        test_dataloader: torch.utils.data.DataLoader,
        train_mode=False,
        visualize_every: int = 1,
        writer: Optional[SummaryWriter] = None,
        step: Optional[int] = 0,
        split="clean",
    ):
        metrics = {}


        filter_indices = [
            70,
            77,
            93,
            96,
            140,
            143,
            162,
            172,
            174,
            179,
            187,
            215,
            236,
            284,
            285,
            293,
            330,
            358,
            368,
            402,
            415,
            458,
            483,
            495,
            534,
        ]

        for ind, sample in enumerate(tqdm(test_dataloader)):
            if ind in filter_indices:
                continue

            if isinstance(sample, tuple):
                sample, gotit = sample
                if not all(gotit):
                    print("batch is None")
                    continue
            if torch.cuda.is_available():
                dataclass_to_cuda_(sample)
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

            if (
                not train_mode
                and hasattr(model, "sequence_len")
                and (sample.visibility[:, : model.sequence_len].sum() == 0)
            ):
                print(f"skipping batch {ind}")
                continue


            flow, flow_alpha = model.forward_flow(video=sample.video, split=split, videodepth=sample.videodepth)
            gt_flow = sample.flow
            gt_alpha = sample.flow_alpha

            out_metrics = compute_cvo_metrics(
                gt={"flow": gt_flow, "alpha": gt_alpha}, pred={"flow": flow, "alpha": flow_alpha}
            )

            metrics[sample.seq_name[0]] = out_metrics
            for metric_name in out_metrics.keys():
                if "avg" not in metrics:
                    metrics["avg"] = {}
                metrics["avg"][metric_name] = float(
                    np.mean([v[metric_name] for k, v in metrics.items() if k != "avg"])
                )


            print("metrics", out_metrics)
        print("avg", metrics["avg"])

        return metrics
