# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from densetrack3d.models.model_utils import (
    bilinear_sample2d,
    bilinear_sampler,
    get_grid,
    get_points_on_a_grid,
    smart_cat,
)
from densetrack3d.utils.io import create_folder, write_frame_np, write_video
from densetrack3d.utils.visualizer import Visualizer, flow_to_rgb
from einops import rearrange, repeat
from tqdm import tqdm


# ind = 0
class EvaluationPredictor2D(torch.nn.Module):
    def __init__(
        self,
        model,
        interp_shape: Tuple[int, int] = (384, 512),
        grid_size: int = 5,
        local_grid_size: int = 8,
        single_point: bool = True,
        n_iters: int = 6,
        use_disp: bool = False,
    ) -> None:
        super(EvaluationPredictor2D, self).__init__()
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

    def forward(self, video, queries, is_sparse=True):
        if is_sparse:
            return self.forward_sparse(video, queries)
        else:
            return self.forward_dense(video, queries)

    def forward_flow2d(self, video, videodepth=None, dst_frame=-1):
        B, T, C, H, W = video.shape
        # device = video.device

        # videodepth_512 = videodepth.clone()
        if H != self.interp_shape[0] or W != self.interp_shape[1]:
            video = F.interpolate(
                video.flatten(0, 1), tuple(self.interp_shape), mode="bilinear", align_corners=True
            ).reshape(B, T, 3, self.interp_shape[0], self.interp_shape[1])
            # videodepth = F.interpolate(videodepth.flatten(0, 1), tuple(self.interp_shape), mode="nearest").reshape(B, T, 1, self.interp_shape[0], self.interp_shape[1])

        sparse_predictions, dense_predictions, _ = self.model(
            video=video,
            # videodepth=videodepth,
            sparse_queries=None,
            iters=self.n_iters,
        )

        dense_traj_e, dense_vis_e = dense_predictions["coords"], dense_predictions["vis"]

        # grid_xy = get_points_on_a_grid((9*4,12*4), video.shape[3:]).long().float()
        # grid_xy = torch.cat([torch.zeros_like(grid_xy[:, :, :1]), grid_xy], dim=2).to(video.device)  # B, N, C

        # traj_e, vis_e, dense_traj_e, dense_vis_e, _, _ = self.model(
        #     video=video,
        #     # videodepth=videodepth,
        #     queries=grid_xy,
        #     iters=self.n_iters,
        # )

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

    def forward_flow(self, video, split="clean", videodepth=None):
        B, T, C, H, W = video.shape
        device = video.device
        # B, N, D = queries.shape

        # assert D == 4

        ori_video = video.clone()

        if H != self.interp_shape[0] or W != self.interp_shape[1]:
            video = video.reshape(B * T, C, H, W)
            video = F.interpolate(video, tuple(self.interp_shape), mode="bilinear", align_corners=True)
            video = video.reshape(B, T, 3, self.interp_shape[0], self.interp_shape[1])

            # breakpoint()
            if videodepth is not None:
                videodepth = videodepth.reshape(B * T, 1, H, W)
                videodepth = F.interpolate(videodepth, tuple(self.interp_shape), mode="nearest")
                videodepth = videodepth.reshape(B, T, 1, self.interp_shape[0], self.interp_shape[1])

        grid_xy = get_points_on_a_grid((9 * 4, 12 * 4), video.shape[3:]).long().float()
        grid_xy = torch.cat([torch.zeros_like(grid_xy[:, :, :1]), grid_xy], dim=2).to(device)  # B, N, C

        src_step = 0

        model_outputs = self.model(
            video=video[:, src_step:],
            # videodepth=videodepth[:, src_step:] if videodepth is not None else None,
            queries=grid_xy,
            iters=self.n_iters,
        )

        traj_e, vis_e, dense_traj_e, dense_vis_e, dense_traj_d_e, _ = model_outputs
        # _, _, dense_traj_e, dense_vis_e, _ = model_outputs

        # # NOTE for vis only:
        # dense_track_reshaped = rearrange(dense_traj_e[0, :, :2], 't c h w -> t h w c')
        # flow_h, flow_w = dense_track_reshaped.shape[1], dense_track_reshaped.shape[2]
        # ori_grid = get_grid(flow_h, flow_w, normalize=False, device=dense_track_reshaped.device)[None]

        # global ind
        # for t in range(T):
        #     # flow = dense_track_reshaped[t:t+1] - ori_grid

        #     # if flow_h != H or flow_w != W:
        #     #     flow = F.interpolate(flow.permute(0,3,1,2), size=(H, W), mode="bilinear").permute(0,2,3,1)
        #     #     flow[..., 0] *= (W - 1) / float(self.interp_shape[1] - 1)
        #     #     flow[..., 1] *= (H - 1) / float(self.interp_shape[0] - 1)

        #     # rgb_flow = flow_to_rgb(flow[0].cpu().numpy())
        #     # # rgb_flow[:, 63:64, 450:455] = np.array([255, 0, 0])[:, None, None]
        #     # # rgb_flow[:, 47:48, 450:455] = np.array([255, 0, 0])[:, None, None]
        #     # # 33, 421
        #     save_folder = f"debug/dense_flow/{ind:05d}"
        #     # os.makedirs(save_folder, exist_ok=True)
        #     # write_frame_np(rgb_flow, os.path.join(save_folder, f"pred_flow_{t}.png"))

        #     print(save_folder)
        #     img = ori_video[0, t].permute(1,2,0).cpu().numpy().astype(np.uint8)
        #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #     cv2.imwrite(
        #         os.path.join(save_folder, f"frame_{t}.png"),
        #         img,
        #     )

        ind = ind + 1

        if split == "clean" or split == "final":
            frame_index = 6
        else:
            frame_index = -1

        dense_track_at_6 = rearrange(dense_traj_e[:, frame_index, :2], "b c h w -> b h w c")

        flow_h, flow_w = dense_track_at_6.shape[1], dense_track_at_6.shape[2]
        ori_grid = get_grid(flow_h, flow_w, normalize=False, device=dense_track_at_6.device).unsqueeze(0)

        # breakpoint()
        flow = dense_track_at_6 - ori_grid
        flow_alpha = dense_vis_e[:, frame_index]  # B 2 H W

        # ori_grid = get_grid(384, 512, normalize=False, device=dense_track_at_6.device).unsqueeze(0)
        # flow = dense_track_at_6 - ori_grid

        if flow_h != H or flow_w != W:
            flow = F.interpolate(flow.permute(0, 3, 1, 2), size=(H, W), mode="bilinear").permute(0, 2, 3, 1)
            flow[..., 0] *= (W - 1) / float(self.interp_shape[1] - 1)
            flow[..., 1] *= (H - 1) / float(self.interp_shape[0] - 1)

            flow_alpha = F.interpolate(flow_alpha.unsqueeze(1), size=(H, W), mode="bilinear")
            flow_alpha = rearrange(flow_alpha, "b 1 h w -> b h w")

        # dense_track_d_at_6 = rearrange(dense_traj_d_e[:, frame_index, :2], 'b c h w -> b h w c')
        # ori_grid_d = get_grid(96, 128, normalize=False, device=dense_track_at_6.device).unsqueeze(0)
        # flow_d = dense_track_d_at_6 - ori_grid_d
        # flow_d = F.interpolate(flow_d.permute(0,3,1,2), size=(H, W), mode="bilinear").permute(0,2,3,1)
        # flow_d[..., 0] *= (W - 1) / float(self.interp_shape[1] - 1)
        # flow_d[..., 1] *= (H - 1) / float(self.interp_shape[0] - 1)

        # dense_track = torch.cat([dense_traj_e, dense_vis_e.float().unsqueeze(2)], dim=2)
        flow_alpha = (flow_alpha > 0.7).float()
        return flow, flow_alpha

    def forward_flow_down(self, video, split="clean"):
        B, T, C, H, W = video.shape
        device = video.device
        # B, N, D = queries.shape

        # assert D == 4

        if H != self.interp_shape[0] or W != self.interp_shape[1]:
            video = video.reshape(B * T, C, H, W)
            video = F.interpolate(video, tuple(self.interp_shape), mode="bilinear", align_corners=True)
            video = video.reshape(B, T, 3, self.interp_shape[0], self.interp_shape[1])

        # depth_init = F.interpolate(depth_init, tuple(self.interp_shape), mode="nearest")

        # tracks = torch.zeros(B, T, N, 4, device=video.device)

        # grid_size = 64
        # grid_xy = get_points_on_a_grid(grid_size, video.shape[3:]).to(device)
        grid_xy = get_points_on_a_grid((9 * 4, 12 * 4), video.shape[3:]).long().float()
        grid_xy = torch.cat([torch.zeros_like(grid_xy[:, :, :1]), grid_xy], dim=2).to(device)  # B, N, C

        src_step = 0

        model_outputs = self.model(
            video=video[:, src_step:],
            # videodepth=videodepth[:, src_step:],
            queries=grid_xy,
            iters=self.n_iters,
        )

        # traj_e, vis_e, dense_traj_e, dense_vis_e, dense_traj_d_e, dense_vis_d_e, _ = model_outputs
        _, _, dense_traj_d_e, dense_vis_d_e, _ = model_outputs

        # breakpoint()
        # dense_traj_e[:, :, 0] *= (W - 1) / float(self.interp_shape[1] - 1)
        # dense_traj_e[:, :, 1] *= (H - 1) / float(self.interp_shape[0] - 1)

        # dense_traj_e[:, :, 0] *= 0.25
        # dense_traj_e[:, :, 1] *= 0.25

        if split == "clean" or split == "final":
            frame_index = 6
        else:
            frame_index = -1
        dense_track_d_at_6 = rearrange(dense_traj_d_e[:, frame_index, :2], "b c h w -> b h w c")

        # ori_grid = get_grid(384, 512, normalize=False, device=dense_track_at_6.device).unsqueeze(0)
        ori_grid_d = get_grid(96, 128, normalize=False, device=dense_track_d_at_6.device).unsqueeze(0)

        flow_d = dense_track_d_at_6 - ori_grid_d

        # flow[..., 0] *= (W - 1) / float(self.interp_shape[1] - 1)
        # flow[..., 1] *= (H - 1) / float(self.interp_shape[0] - 1)
        # breakpoint()
        flow_d = F.interpolate(flow_d.permute(0, 3, 1, 2), size=(H // 4, W // 4), mode="bilinear").permute(0, 2, 3, 1)
        flow_d[..., 0] *= (W // 4 - 1) / float(self.interp_shape[1] // 4 - 1)
        flow_d[..., 1] *= (H // 4 - 1) / float(self.interp_shape[0] // 4 - 1)

        flow_alpha = dense_vis_d_e[:, frame_index]  # B 2 H W
        flow_alpha = F.interpolate(flow_alpha.unsqueeze(1), size=(H // 4, W // 4), mode="bilinear")
        flow_alpha = rearrange(flow_alpha, "b 1 h w -> b h w")

        # dense_track = torch.cat([dense_traj_e, dense_vis_e.float().unsqueeze(2)], dim=2)
        flow_alpha = (flow_alpha > 0.5).float()
        return flow_d, flow_alpha

    def forward_dense(self, video, queries):
        queries = queries.clone()
        B, T, C, H, W = video.shape
        B, N, D = queries.shape

        # assert D == 4

        video = video.reshape(B * T, C, H, W)
        video = F.interpolate(video, tuple(self.interp_shape), mode="bilinear", align_corners=True)
        video = video.reshape(B, T, 3, self.interp_shape[0], self.interp_shape[1])

        # depth_init = F.interpolate(depth_init, tuple(self.interp_shape), mode="nearest")

        # videodepth = F.interpolate(
        #     videodepth.reshape(B * T, 1, H, W),
        #     tuple(self.interp_shape), mode="nearest"
        # ).reshape(B, T, 1, self.interp_shape[0], self.interp_shape[1])

        device = video.device

        tracks = torch.zeros(B, T, N, 3, device=video.device)

        grid_size = 32
        # grid_xy = get_points_on_a_grid(grid_size, video.shape[3:]).to(device)
        grid_xy = get_points_on_a_grid(grid_size, video.shape[3:]).long().float()
        grid_xy = torch.cat([torch.zeros_like(grid_xy[:, :, :1]), grid_xy], dim=2).to(device)  # B, N, C

        # queries[:, :, 1] *= (self.interp_shape[1] - 1) / (W - 1)
        # queries[:, :, 2] *= (self.interp_shape[0] - 1) / (H - 1)

        src_steps = [int(v) for v in torch.unique(queries[..., 0])]
        # src_steps = [0]
        for src_step in tqdm(src_steps, desc="Refine source step", leave=False):
            # grid_xy_f = src_step * torch.ones_like(grid_xy[:, :, :1])
            # xy_d = bilinear_sample2d(depth_init[:1], xy[..., 0], xy[..., 1]).to(device)
            # breakpoint()
            # grid_xy_d = bilinear_sampler(videodepth[:, src_step], rearrange(grid_xy[..., 1:3], 'b n c -> b () n c'), mode="nearest")
            # grid_xy_d = rearrange(grid_xy_d, 'b c m n -> b (m n) c')

            # grid_queries = torch.cat([grid_xy, grid_xy_d], dim=2)  #

            # new_queries = torch.cat([queries, new_queries], dim=1)
            # new_queries = queries

            grid_queries = grid_xy

            model_outputs = self.model(
                video=video[:, src_step:],
                # videodepth=videodepth[:, src_step:],
                queries=grid_queries,
                iters=self.n_iters,
            )

            # traj_e, vis_e, dense_traj_e, dense_vis_e, _ = model_outputs
            _, _, dense_traj_e, dense_vis_e, _, _ = model_outputs

            # breakpoint()
            # traj_e[:, :, :, 0] *= (W - 1) / float(self.interp_shape[1] - 1)
            # traj_e[:, :, :, 1] *= (H - 1) / float(self.interp_shape[0] - 1)

            dense_traj_e[:, :, 0] *= (W - 1) / float(self.interp_shape[1] - 1)
            dense_traj_e[:, :, 1] *= (H - 1) / float(self.interp_shape[0] - 1)

            dense_vis_e = (dense_vis_e > 0.8).float()

            dense_track = torch.cat([dense_traj_e, dense_vis_e.float().unsqueeze(2)], dim=2)

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
                    # tracks[b, src_step, cur, 3] = cur_points[:, 3]

        # breakpoint()
        traj_e = tracks[..., :2]
        vis_e = tracks[..., 2]

        return traj_e, vis_e

    def dense_to_sparse_tracks_3d(self, x, y, tracks):
        # h, w = height, width
        T = tracks.size(0)
        grid = torch.stack([x, y], dim=-1) * 2 - 1
        grid = repeat(grid, "s c -> t s r c", t=T, r=1)

        # tracks = rearrange(tracks, "t h w c -> t c h w")
        # tracks2d = torch.cat([tracks[:, :2, ...], tracks[:, 3:, ...]], dim=1)
        tracks2d = tracks
        # tracks_d = tracks[:, 2:3, ...]
        tracks2d = F.grid_sample(tracks2d, grid, align_corners=True, mode="bilinear")
        tracks2d = rearrange(tracks2d[..., 0], "t c s -> t s c")
        # tracks2d[..., 0] = tracks2d[..., 0]
        # tracks2d[..., 1] = tracks2d[..., 1] * (h - 1)
        tracks2d[..., 2] = (tracks2d[..., 2] > 0).float()

        # tracks_d = F.grid_sample(tracks_d, grid, align_corners=True, mode="nearest")
        # tracks_d = rearrange(tracks_d[..., 0], "t c s -> t s c")
        # tracks[..., 3] = tracks[..., 3] * (h - 1)

        # breakpoint()
        return tracks2d
        # return torch.cat([tracks2d, tracks_d], dim=-1)

    def forward_sparse(self, video, queries):
        queries = queries.clone()
        B, T, C, H, W = video.shape
        B, N, D = queries.shape

        # assert D == 4

        video = video.reshape(B * T, C, H, W)
        video = F.interpolate(video, tuple(self.interp_shape), mode="bilinear", align_corners=True)
        video = video.reshape(B, T, 3, self.interp_shape[0], self.interp_shape[1])

        device = video.device

        queries[:, :, 1] *= (self.interp_shape[1] - 1) / (W - 1)
        queries[:, :, 2] *= (self.interp_shape[0] - 1) / (H - 1)

        if self.single_point:
            traj_e = torch.zeros((B, T, N, 2), device=device)
            vis_e = torch.zeros((B, T, N), device=device)
            for pind in range((N)):
                query = queries[:, pind : pind + 1]

                t = query[0, 0, 0].long()

                traj_e_pind, vis_e_pind = self._process_one_point(video, query)
                # breakpoint()
                if traj_e_pind.shape[-1] == 3:  # NOTE 3D TRAJECTORIES
                    traj_e_pind = traj_e_pind[..., :2]
                traj_e[:, t:, pind : pind + 1] = traj_e_pind[:, :, :1]
                vis_e[:, t:, pind : pind + 1] = vis_e_pind[:, :, :1]
        else:
            if self.grid_size > 0:
                xy = get_points_on_a_grid(self.grid_size, video.shape[3:])
                xy = torch.cat([torch.zeros_like(xy[:, :, :1]), xy], dim=2).to(device)  #

                # xy_d = bilinear_sample2d(depth_init[:1], xy[..., 1], xy[..., 2])
                # xy = torch.cat([xy, xy_d.permute(0,2,1)], dim=2).to(device)  #

                queries = torch.cat([queries, xy], dim=1)  #

            model_outputs = self.model(
                video=video,
                queries=queries,
                iters=self.n_iters,
                use_dense=False,
            )

            if len(model_outputs) == 3:
                traj_e, vis_e, _ = model_outputs
            else:
                traj_e, vis_e, _, _, _, _ = model_outputs

        traj_e[:, :, :, 0] *= (W - 1) / float(self.interp_shape[1] - 1)
        traj_e[:, :, :, 1] *= (H - 1) / float(self.interp_shape[0] - 1)
        return traj_e, vis_e

    def forward_lift3D(self, video, videodepth, queries, depth_init, return_3d=False, is_sparse=True):
        queries = queries.clone()

        queries = queries[..., :3]  # NOTE remove depth

        B, T, C, H, W = video.shape
        B, N, D = queries.shape

        # assert D == 4

        video = video.reshape(B * T, C, H, W)
        video = F.interpolate(video, tuple(self.interp_shape), mode="bilinear", align_corners=True)
        video = video.reshape(B, T, 3, self.interp_shape[0], self.interp_shape[1])

        depth_init = F.interpolate(depth_init, tuple(self.interp_shape), mode="nearest")

        videodepth = F.interpolate(
            videodepth.reshape(B * T, 1, H, W), tuple(self.interp_shape), mode="nearest"
        ).reshape(B, T, 1, self.interp_shape[0], self.interp_shape[1])

        device = video.device

        queries[:, :, 1] *= (self.interp_shape[1] - 1) / (W - 1)
        queries[:, :, 2] *= (self.interp_shape[0] - 1) / (H - 1)

        if self.single_point:
            traj_e = torch.zeros((B, T, N, 2), device=device)
            vis_e = torch.zeros((B, T, N), device=device)
            for pind in range((N)):
                query = queries[:, pind : pind + 1]

                t = query[0, 0, 0].long()

                traj_e_pind, vis_e_pind = self._process_one_point(video, query)
                # breakpoint()
                if traj_e_pind.shape[-1] == 3:  # NOTE 3D TRAJECTORIES
                    traj_e_pind = traj_e_pind[..., :2]
                traj_e[:, t:, pind : pind + 1] = traj_e_pind[:, :, :1]
                vis_e[:, t:, pind : pind + 1] = vis_e_pind[:, :, :1]
        else:
            if self.grid_size > 0:
                xy = get_points_on_a_grid(self.grid_size, video.shape[3:])
                xy = torch.cat([torch.zeros_like(xy[:, :, :1]), xy], dim=2).to(device)  #

                # xy_d = bilinear_sample2d(depth_init[:1], xy[..., 1], xy[..., 2])
                # xy = torch.cat([xy, xy_d.permute(0,2,1)], dim=2).to(device)  #

                queries = torch.cat([queries, xy], dim=1)  #

            model_outputs = self.model(
                video=video,
                queries=queries,
                iters=self.n_iters,
                use_dense=False,
            )

            if len(model_outputs) == 3:
                traj_e, vis_e, _ = model_outputs
            else:
                traj_e, vis_e, _, _, _, _ = model_outputs

        # traj_e: B T N 2
        traj_d_e = []
        for t in range(T):
            traj_d_ = bilinear_sample2d(videodepth[:, t], traj_e[:, t, :, 0], traj_e[:, t, :, 1])
            traj_d_ = traj_d_.permute(0, 2, 1)

            traj_d_e.append(traj_d_)
        traj_d_e = torch.stack(traj_d_e, dim=1)  # B T N 1

        # breakpoint()

        # print("debug", traj_d_e.shape, traj_e.shape)
        traj_e[:, :, :, 0] *= (W - 1) / float(self.interp_shape[1] - 1)
        traj_e[:, :, :, 1] *= (H - 1) / float(self.interp_shape[0] - 1)

        if return_3d:

            traj_3d = torch.cat([traj_e, traj_d_e], dim=-1)
            return traj_3d, vis_e
        return traj_e, vis_e

    def forward_3dflow(self, video, videodepth):

        grid_size = 80
        B, T, _, H, W = video.shape
        grid_step = W // grid_size
        grid_width = W // grid_step
        grid_height = H // grid_step
        # tracks = visibilities = None

        dense_tracks = torch.zeros((B, T, 3, H, W), dtype=torch.float32).to(video.device)
        dense_visibilities = torch.zeros((B, T, H, W), dtype=torch.float32).to(video.device)

        grid_pts = torch.zeros((1, grid_width * grid_height, 3)).to(video.device)
        grid_pts[0, :, 0] = 0

        for offset in range(grid_step * grid_step):
            print(f"step {offset} / {grid_step * grid_step}")
            ox = offset % grid_step
            oy = offset // grid_step
            grid_pts[0, :, 1] = torch.arange(grid_width).repeat(grid_height) * grid_step + ox
            grid_pts[0, :, 2] = torch.arange(grid_height).repeat_interleave(grid_width) * grid_step + oy

            # grid_pts[0, :, 3:4] = bilinear_sample2d(depths_init[:1], grid_pts[..., 1], grid_pts[..., 2]).permute(0,2,1)

            n_ori_queries = grid_pts.shape[1]

            preds = self.forward_lift3D(
                video=video,
                videodepth=videodepth,
                queries=grid_pts,
                depth_init=videodepth[:, 0],
                return_3d=True,
                is_sparse=True,
            )
            pred_tracks, pred_visibilities = preds

            pred_tracks = pred_tracks[:, :, :n_ori_queries]
            pred_visibilities = pred_visibilities[:, :, :n_ori_queries]

            valid_mask = (
                (grid_pts[0, :, 2].long() < H)
                & (grid_pts[0, :, 1].long() < W)
                & (grid_pts[0, :, 2].long() >= 0)
                & (grid_pts[0, :, 1].long() >= 0)
            )
            valid_y = grid_pts[0, :, 2].long()[valid_mask]
            valid_x = grid_pts[0, :, 1].long()[valid_mask]
            dense_tracks[:, :, :, valid_y, valid_x] = rearrange(pred_tracks[:, :, valid_mask], "b t n d -> b t d n")
            dense_visibilities[:, :, valid_y, valid_x] = pred_visibilities[:, :, valid_mask]

            # print(tracks_step.shape, visibilities_step.shape)
            # tracks = smart_cat(tracks, tracks_step, dim=2)
            # visibilities = smart_cat(visibilities, visibilities_step, dim=2)

            # if offset >= 0:
            #     break

        return dense_tracks, dense_visibilities

        # B, T, C, H, W = video.shape
        # device = video.device

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

            xy_target_d = bilinear_sample2d(depth_init[:1], xy_target[..., 1], xy_target[..., 2])
            xy_target = torch.cat([xy_target, xy_target_d.permute(0, 2, 1)], dim=2).to(device)  #

            query = torch.cat([query, xy_target], dim=1)  #

        if self.grid_size > 0:
            xy = get_points_on_a_grid(self.grid_size, video.shape[3:])
            xy = torch.cat([torch.zeros_like(xy[:, :, :1]), xy], dim=2).to(device)  #

            xy_d = bilinear_sample2d(depth_init[:1], xy[..., 1], xy[..., 2])
            xy = torch.cat([xy, xy_d.permute(0, 2, 1)], dim=2).to(device)  #

            query = torch.cat([query, xy], dim=1)  #
        # crop the video to start from the queried frame
        query[0, 0, 0] = 0

        traj_e_pind, vis_e_pind, __ = self.model(
            video=video[:, t:], videodepth=videodepth[:, t:], queries=query, iters=self.n_iters
        )

        return traj_e_pind, vis_e_pind
