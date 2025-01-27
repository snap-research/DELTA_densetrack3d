# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
import torch.nn.functional as F
from densetrack3d.models.densetrack3d.densetrack3d import DenseTrack3D
from densetrack3d.models.model_utils import (
    bilinear_sample2d,
    bilinear_sampler,
    depth_to_disparity,
    disparity_to_depth,
    get_grid,
    get_points_on_a_grid,
    smart_cat,
)
from einops import rearrange, repeat
from tqdm import tqdm


class EvaluationPredictor(torch.nn.Module):
    def __init__(
        self,
        model: DenseTrack3D,
        interp_shape: Tuple[int, int] = (384, 512),
        grid_size: int = 5,
        local_grid_size: int = 8,
        single_point: bool = True,
        n_iters: int = 6,
    ) -> None:
        super(EvaluationPredictor, self).__init__()

        if isinstance(grid_size, int):
            grid_size = (grid_size, grid_size)
        self.grid_size = grid_size
        self.local_grid_size = local_grid_size
        self.single_point = single_point
        self.interp_shape = interp_shape
        self.n_iters = n_iters

        self.model = model
        self.model.eval()

        ori_grid = get_grid(self.interp_shape[0], self.interp_shape[1], normalize=False)
        ori_grid = rearrange(ori_grid, "h w c -> 1 c h w")

        self.register_buffer("ori_grid", ori_grid, persistent=False)

    def forward(
        self,
        video,
        videodepth,
        queries,
        depth_init,
        intrs=None,
        is_sparse=True,
        return_3d=False,
        lift_3d=False,
        use_2d_only=False,
    ):
        if is_sparse:
            return self.forward_sparse(
                video,
                videodepth,
                queries,
                depth_init,
                intrs=intrs,
                return_3d=return_3d,
                lift_3d=lift_3d,
                use_2d_only=use_2d_only,
            )
        else:
            return self.forward_dense(
                video, videodepth, queries, depth_init, intrs=intrs, return_3d=return_3d, lift_3d=lift_3d
            )

    def forward_flow2d(self, video, videodepth, dst_frame=-1):
        B, T, C, H, W = video.shape
        # device = video.device

        # videodepth_512 = videodepth.clone()
        if H != self.interp_shape[0] or W != self.interp_shape[1]:
            video = F.interpolate(
                video.flatten(0, 1), tuple(self.interp_shape), mode="bilinear", align_corners=True
            ).reshape(B, T, 3, self.interp_shape[0], self.interp_shape[1])
            videodepth = F.interpolate(videodepth.flatten(0, 1), tuple(self.interp_shape), mode="nearest").reshape(
                B, T, 1, self.interp_shape[0], self.interp_shape[1]
            )

        sparse_predictions, dense_predictions, _ = self.model(
            video=video,
            videodepth=videodepth,
            sparse_queries=None,
            iters=self.n_iters,
        )

        dense_traj_e, dense_traj_d_e, dense_vis_e = (
            dense_predictions["coords"],
            dense_predictions["coord_depths"],
            dense_predictions["vis"],
        )

        flow_to_last = dense_traj_e[:, dst_frame, :2] - self.ori_grid
        flow_alpha = dense_vis_e[:, dst_frame]  # B 2 H W

        # TODO find a better way to handle this instead of interpolate
        if self.interp_shape[0] != H or self.interp_shape[1] != W:
            flow_to_last = F.interpolate(flow_to_last, size=(H, W), mode="bilinear")
            flow_to_last[:, 0] = flow_to_last[:, 0] * (W - 1) / float(self.interp_shape[1] - 1)
            flow_to_last[:, 1] = flow_to_last[:, 1] * (H - 1) / float(self.interp_shape[0] - 1)

            flow_alpha = F.interpolate(flow_alpha.unsqueeze(1), size=(H, W), mode="bilinear")

        flow_to_last = rearrange(flow_to_last, "b c h w -> b h w c")
        flow_alpha = rearrange(flow_alpha, "b 1 h w -> b h w")
        flow_alpha = (flow_alpha > 0.9).float()

        return flow_to_last, flow_alpha

    def forward_flow3d(self, video, videodepth):
        B, T, C, H, W = video.shape

        if H != self.interp_shape[0] or W != self.interp_shape[1]:
            video = F.interpolate(
                video.flatten(0, 1), tuple(self.interp_shape), mode="bilinear", align_corners=True
            ).reshape(B, T, 3, self.interp_shape[0], self.interp_shape[1])
            videodepth = F.interpolate(videodepth.flatten(0, 1), tuple(self.interp_shape), mode="nearest").reshape(
                B, T, 1, self.interp_shape[0], self.interp_shape[1]
            )

        sparse_predictions, dense_predictions, _ = self.model(
            video=video,
            videodepth=videodepth,
            sparse_queries=None,
            iters=self.n_iters,
        )

        dense_traj_e, dense_traj_d_e, dense_vis_e = (
            dense_predictions["coords"],
            dense_predictions["coord_depths"],
            dense_predictions["vis"],
        )

        # _, _, _, dense_traj_e, _, dense_vis_e, _, _ = model_outputs

        dense_traj_uvd = torch.cat([dense_traj_e, dense_traj_d_e], dim=2)
        dense_vis_e = dense_vis_e > 0.9

        return dense_traj_uvd, dense_vis_e

    def forward_flow_down(self, video, videodepth):
        B, T, C, H, W = video.shape
        device = video.device
        # B, N, D = queries.shape

        # assert D == 4

        if H != self.interp_shape[0] or W != self.interp_shape[1]:
            video = video.reshape(B * T, C, H, W)
            video = F.interpolate(video, tuple(self.interp_shape), mode="bilinear", align_corners=True)
            video = video.reshape(B, T, 3, self.interp_shape[0], self.interp_shape[1])

            videodepth = F.interpolate(
                videodepth.reshape(B * T, 1, H, W), tuple(self.interp_shape), mode="nearest"
            ).reshape(B, T, 1, self.interp_shape[0], self.interp_shape[1])

        # tracks = torch.zeros(B, T, N, 4, device=video.device)

        src_step = 0
        # grid_size = 64
        # grid_xy = get_points_on_a_grid(grid_size, video.shape[3:]).to(device)
        grid_xy = get_points_on_a_grid((9 * 4, 12 * 4), video.shape[3:]).long().float()
        grid_xy = torch.cat([torch.zeros_like(grid_xy[:, :, :1]), grid_xy], dim=2).to(device)  # B, N, C

        grid_xy_d = bilinear_sampler(
            videodepth[:, src_step], rearrange(grid_xy[..., 1:3], "b n c -> b () n c"), mode="nearest"
        )
        grid_xy_d = rearrange(grid_xy_d, "b c m n -> b (m n) c")

        grid_queries = torch.cat([grid_xy, grid_xy_d], dim=2)  #

        model_outputs = self.model(
            video=video[:, src_step:],
            videodepth=videodepth[:, src_step:],
            queries=grid_queries,
            iters=self.n_iters,
        )

        # traj_e, vis_e, dense_traj_e, dense_vis_e, dense_traj_d_e, dense_vis_d_e, _ = model_outputs
        _, _, dense_traj_d_e, dense_vis_d_e, _ = model_outputs

        dense_track_d_at_6 = rearrange(dense_traj_d_e[:, 6, :2], "b c h w -> b h w c")

        # ori_grid = get_grid(384, 512, normalize=False, device=dense_track_at_6.device).unsqueeze(0)
        ori_grid_d = get_grid(96, 128, normalize=False, device=dense_track_d_at_6.device).unsqueeze(0)

        flow_d = dense_track_d_at_6 - ori_grid_d

        # flow[..., 0] *= (W - 1) / float(self.interp_shape[1] - 1)
        # flow[..., 1] *= (H - 1) / float(self.interp_shape[0] - 1)
        # breakpoint()
        flow_d = F.interpolate(flow_d.permute(0, 3, 1, 2), size=(H // 4, W // 4), mode="bilinear").permute(0, 2, 3, 1)
        flow_d[..., 0] *= (W // 4 - 1) / float(self.interp_shape[1] // 4 - 1)
        flow_d[..., 1] *= (H // 4 - 1) / float(self.interp_shape[0] // 4 - 1)

        flow_alpha = dense_vis_d_e[:, 6]  # B 2 H W
        flow_alpha = F.interpolate(flow_alpha.unsqueeze(1), size=(H // 4, W // 4), mode="bilinear")
        flow_alpha = rearrange(flow_alpha, "b 1 h w -> b h w")

        # dense_track = torch.cat([dense_traj_e, dense_vis_e.float().unsqueeze(2)], dim=2)
        flow_alpha = (flow_alpha > 0.5).float()
        return flow_d, flow_alpha

    def forward_dense(self, video, videodepth, queries, depth_init, return_3d=False):
        queries = queries.clone()
        B, T, C, H, W = video.shape
        B, N, D = queries.shape

        assert D == 4

        video = video.reshape(B * T, C, H, W)
        video = F.interpolate(video, tuple(self.interp_shape), mode="bilinear", align_corners=True)
        video = video.reshape(B, T, 3, self.interp_shape[0], self.interp_shape[1])

        # depth_init = F.interpolate(depth_init, tuple(self.interp_shape), mode="nearest")

        videodepth = F.interpolate(
            videodepth.reshape(B * T, 1, H, W), tuple(self.interp_shape), mode="nearest"
        ).reshape(B, T, 1, self.interp_shape[0], self.interp_shape[1])

        device = video.device

        tracks = torch.zeros(B, T, N, 4, device=video.device)

        # grid_size = 32
        # grid_xy = get_points_on_a_grid(grid_size, video.shape[3:]).to(device)
        grid_xy = get_points_on_a_grid((12, 16), video.shape[3:]).long().float()
        grid_xy = torch.cat([torch.zeros_like(grid_xy[:, :, :1]), grid_xy], dim=2).to(device)  # B, N, C

        # queries[:, :, 1] *= (self.interp_shape[1] - 1) / (W - 1)
        # queries[:, :, 2] *= (self.interp_shape[0] - 1) / (H - 1)

        src_steps = [int(v) for v in torch.unique(queries[..., 0])]
        # src_steps = [0]
        for src_step in tqdm(src_steps, desc="Refine source step", leave=False):
            # grid_xy_f = src_step * torch.ones_like(grid_xy[:, :, :1])
            # xy_d = bilinear_sample2d(depth_init[:1], xy[..., 0], xy[..., 1]).to(device)
            # breakpoint()
            grid_xy_d = bilinear_sampler(
                videodepth[:, src_step], rearrange(grid_xy[..., 1:3], "b n c -> b () n c"), mode="nearest"
            )
            grid_xy_d = rearrange(grid_xy_d, "b c m n -> b (m n) c")

            grid_queries = torch.cat([grid_xy, grid_xy_d], dim=2)  #

            # new_queries = torch.cat([queries, new_queries], dim=1)
            # new_queries = queries

            model_outputs = self.model(
                video=video[:, src_step:],
                videodepth=videodepth[:, src_step:],
                queries=grid_queries,
                iters=self.n_iters,
            )

            traj_e, traj_d_e, vis_e, dense_traj_e, dense_traj_d_e, dense_vis_e, _, _ = model_outputs

            # breakpoint()
            # traj_e[:, :, :, 0] *= (W - 1) / float(self.interp_shape[1] - 1)
            # traj_e[:, :, :, 1] *= (H - 1) / float(self.interp_shape[0] - 1)

            dense_traj_e[:, :, 0] *= (W - 1) / float(self.interp_shape[1] - 1)
            dense_traj_e[:, :, 1] *= (H - 1) / float(self.interp_shape[0] - 1)

            dense_track = torch.cat([dense_traj_e, dense_traj_d_e, dense_vis_e.float().unsqueeze(2)], dim=2)

            # dense_traj_e: B T C H W
            # x_grid = queries[0,]
            for b in range(B):
                cur = queries[b, :, 0] == src_step
                if torch.any(cur):
                    cur_points = queries[b, cur].clone()
                    cur_x = cur_points[..., 1] / float(W - 1)
                    cur_y = cur_points[..., 2] / float(H - 1)
                    # cur_x = cur_points[..., 1] / float(self.interp_shape[1] - 1)
                    # cur_y = cur_points[..., 2] / float(self.interp_shape[0] - 1)
                    cur_tracks = self.dense_to_sparse_tracks_3d(cur_x, cur_y, dense_track[b])
                    tracks[b, src_step:, cur] = cur_tracks

                    # NOTE override query points
                    tracks[b, src_step, cur, :2] = cur_points[:, 1:3]
                    tracks[b, src_step, cur, 2] = 1
                    tracks[b, src_step, cur, 3] = cur_points[:, 3]

        # breakpoint()
        traj_e = torch.cat([tracks[..., :2], tracks[..., 3:]], dim=-1)
        vis_e = tracks[..., 2]

        return traj_e, vis_e

    def dense_to_sparse_tracks_3d(self, x, y, tracks):
        # h, w = height, width
        T = tracks.size(0)
        grid = torch.stack([x, y], dim=-1) * 2 - 1
        grid = repeat(grid, "s c -> t s r c", t=T, r=1)

        # tracks = rearrange(tracks, "t h w c -> t c h w")
        tracks2d = torch.cat([tracks[:, :2, ...], tracks[:, 3:, ...]], dim=1)
        tracks_d = tracks[:, 2:3, ...]
        tracks2d = F.grid_sample(tracks2d, grid, align_corners=True, mode="bilinear")
        tracks2d = rearrange(tracks2d[..., 0], "t c s -> t s c")
        # tracks2d[..., 0] = tracks2d[..., 0]
        # tracks2d[..., 1] = tracks2d[..., 1] * (h - 1)
        # tracks2d[..., 2] = (tracks2d[..., 2] > 0).float()

        tracks_d = F.grid_sample(tracks_d, grid, align_corners=True, mode="nearest")
        tracks_d = rearrange(tracks_d[..., 0], "t c s -> t s c")
        # tracks[..., 3] = tracks[..., 3] * (h - 1)

        # breakpoint()

        return torch.cat([tracks2d, tracks_d], dim=-1)

    def forward_sparse(
        self, video, videodepth, queries, depth_init, intrs=None, return_3d=False, lift_3d=False, use_2d_only=False
    ):

        device = video.device
        queries = queries.clone()
        B, T, C, H, W = video.shape
        B, N, D = queries.shape

        assert D == 4

        depth_init = F.interpolate(depth_init, tuple(self.interp_shape), mode="nearest")
        video = F.interpolate(
            video.flatten(0, 1), tuple(self.interp_shape), mode="bilinear", align_corners=True
        ).reshape(B, T, 3, self.interp_shape[0], self.interp_shape[1])
        videodepth = F.interpolate(videodepth.flatten(0, 1), tuple(self.interp_shape), mode="nearest").reshape(
            B, T, 1, self.interp_shape[0], self.interp_shape[1]
        )

        queries[:, :, 1] *= (self.interp_shape[1] - 1) / (W - 1)
        queries[:, :, 2] *= (self.interp_shape[0] - 1) / (H - 1)

        num_ori_queries = queries.shape[1]

        if self.grid_size[0] > 0:
            xy = get_points_on_a_grid(self.grid_size, video.shape[3:])
            xy = torch.cat([torch.zeros_like(xy[:, :, :1]), xy], dim=2).to(device)  #

            xy_d = bilinear_sample2d(depth_init[:1], xy[..., 1], xy[..., 2])
            xy = torch.cat([xy, xy_d.permute(0, 2, 1)], dim=2).to(device)  #

            queries = torch.cat([queries, xy], dim=1)  #

        if use_2d_only:
            sparse_predictions, dense_predictions, _ = self.model(
                video=video,
                # videodepth=videodepth,
                sparse_queries=queries,
                iters=self.n_iters,
                use_dense=False,
                # n_real_queries=num_ori_queries,
            )
            traj_e, vis_e = sparse_predictions["coords"], sparse_predictions["vis"]

            traj_d_e = []
            for t in range(T):
                traj_d_ = bilinear_sample2d(videodepth[:, t], traj_e[:, t, :, 0], traj_e[:, t, :, 1])
                traj_d_ = traj_d_.permute(0, 2, 1)

                traj_d_e.append(traj_d_)
            traj_d_e = torch.stack(traj_d_e, dim=1)  # B T N 1
        else:
            sparse_predictions, dense_predictions, _ = self.model(
                video=video,
                videodepth=videodepth,
                sparse_queries=queries,
                iters=self.n_iters,
                use_dense=False,
                # n_real_queries=num_ori_queries,
            )
            traj_e, traj_d_e, vis_e = (
                sparse_predictions["coords"],
                sparse_predictions["coord_depths"],
                sparse_predictions["vis"],
            )

        if lift_3d and not use_2d_only:
            traj_d_e = []
            for t in range(T):
                traj_d_ = bilinear_sample2d(videodepth[:, t], traj_e[:, t, :, 0], traj_e[:, t, :, 1])
                traj_d_ = traj_d_.permute(0, 2, 1)

                traj_d_e.append(traj_d_)
            traj_d_e = torch.stack(traj_d_e, dim=1)  # B T N 1

        traj_e[:, :, :, 0] *= (W - 1) / float(self.interp_shape[1] - 1)
        traj_e[:, :, :, 1] *= (H - 1) / float(self.interp_shape[0] - 1)

        return traj_e, traj_d_e, vis_e

    def _process_one_point(self, video, videodepth, query, depth_init):
        t = query[0, 0, 0].long()

        device = query.device
        if self.local_grid_size > 0:
            xy_target = get_points_on_a_grid(
                self.local_grid_size,
                (50, 50),
                [query[0, 0, 2].item(), query[0, 0, 1].item()],
            )

            xy_target = torch.cat([torch.zeros_like(xy_target[:, :, :1]), xy_target], dim=2).to(device)  #

            xy_target_d = bilinear_sample2d(videodepth[:, t], xy_target[..., 1], xy_target[..., 2])
            xy_target = torch.cat([xy_target, xy_target_d.permute(0, 2, 1)], dim=2).to(device)  #

            query = torch.cat([query, xy_target], dim=1)  #

        # if self.grid_size[0] > 0:
        #     xy = get_points_on_a_grid(self.grid_size, video.shape[3:])
        #     xy = torch.cat([torch.zeros_like(xy[:, :, :1]), xy], dim=2).to(device)  #

        #     xy_d = bilinear_sample2d(depth_init[:1], xy[..., 1], xy[..., 2])
        #     xy = torch.cat([xy, xy_d.permute(0,2,1)], dim=2).to(device)  #

        #     query = torch.cat([query, xy], dim=1)  #

        if self.grid_size[0] > 0:
            xy = get_points_on_a_grid(self.grid_size, video.shape[3:])
            xy = torch.cat([torch.zeros_like(xy[:, :, :1]), xy], dim=2).to(device)  #

            xy_d = bilinear_sample2d(videodepth[:, t], xy[..., 1], xy[..., 2])
            xy = torch.cat([xy, xy_d.permute(0, 2, 1)], dim=2).to(device)  #

            query = torch.cat([query, xy], dim=1)  #
        # crop the video to start from the queried frame
        query[0, 0, 0] = 0
        # breakpoint()
        # print(query.shape)
        model_outputs = self.model(
            video=video[:, t:],
            videodepth=videodepth[:, t:],
            queries=query,
            iters=self.n_iters,
            use_dense=False,
        )

        if len(model_outputs) == 3:
            traj_e, vis_e, _ = model_outputs
        elif len(model_outputs) == 10:
            traj_e, traj_d_e, vis_e, dense_traj_e, dense_traj_d_e, dense_vis_e, _, _, _, _ = model_outputs
        else:
            traj_e, traj_d_e, vis_e, dense_traj_e, dense_traj_d_e, dense_vis_e, _, _ = model_outputs

        # print(vis_e.shape, vis_e.shape)
        return traj_e, vis_e
