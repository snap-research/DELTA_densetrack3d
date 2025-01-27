# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# import pytorch3d.ops as pt3dops
import torch
import torch.nn.functional as F

from densetrack3d.models.model_utils import (
    bilinear_sample2d,
    bilinear_sampler,
    get_grid,
    get_points_on_a_grid,
    smart_cat,
)
from einops import rearrange
from tqdm import tqdm








class CoTrackerDensePredictor(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.support_grid_size = 6

        self.interp_shape = model.model_resolution
        self.model = model
        # self.model.eval()

        self.n_iters = 6

    @torch.no_grad()
    def forward(
        self,
        video,  # (B, T, 3, H, W)
        videodepth,
        grid_query_frame: int = 0,  # only for dense and regular grid tracks
        backward_tracking: bool = False,
        return_virtual_tokens: bool = False,
        prev_virtual_tokens: list = None,
        cond_trajs: torch.Tensor = None,
    ):
        B, T, C, H, W = video.shape

        ori_video = video.clone()
        ori_videodepth = videodepth.clone()

        video = video.reshape(B * T, C, H, W)
        video = F.interpolate(video, tuple(self.interp_shape), mode="bilinear", align_corners=True)
        video = video.reshape(B, T, 3, self.interp_shape[0], self.interp_shape[1])

        videodepth = F.interpolate(
            videodepth.reshape(B * T, 1, H, W), tuple(self.interp_shape), mode="nearest"
        ).reshape(B, T, 1, self.interp_shape[0], self.interp_shape[1])

        grid_size = 80
        B, T = video.shape[:2]
        grid_step = self.interp_shape[1] // grid_size
        grid_width = self.interp_shape[1] // grid_step
        grid_height = self.interp_shape[0] // grid_step
        # tracks = visibilities = None

        dense_tracks = torch.zeros((B, T, 3, self.interp_shape[0], self.interp_shape[1]), dtype=torch.float32).to(
            video.device
        )
        dense_visibilities = torch.zeros((B, T, self.interp_shape[0], self.interp_shape[1]), dtype=torch.float32).to(
            video.device
        )

        grid_pts = torch.zeros((1, grid_width * grid_height, 3)).to(video.device)
        grid_pts[0, :, 0] = 0

        for offset in range(grid_step * grid_step):
            # for offset in range(2): # for DEBUG
            print(f"step {offset} / {grid_step * grid_step}")
            ox = offset % grid_step
            oy = offset // grid_step
            grid_pts[0, :, 1] = torch.arange(grid_width).repeat(grid_height) * grid_step + ox
            grid_pts[0, :, 2] = torch.arange(grid_height).repeat_interleave(grid_width) * grid_step + oy

            valid_mask = (
                (grid_pts[0, :, 2].long() < H)
                & (grid_pts[0, :, 1].long() < W)
                & (grid_pts[0, :, 2].long() >= 0)
                & (grid_pts[0, :, 1].long() >= 0)
            )
            valid_y = grid_pts[0, :, 2].long()[valid_mask]
            valid_x = grid_pts[0, :, 1].long()[valid_mask]
            grid_pts = grid_pts[:, valid_mask]

            # grid_pts[0, :, 3:4] = bilinear_sample2d(depths_init[:1], grid_pts[..., 1], grid_pts[..., 2]).permute(0,2,1)

            # grid_pts[..., 0] *= (W - 1) / float(self.interp_shape[1] - 1)
            # grid_pts[..., 1] *= (H - 1) / float(self.interp_shape[0] - 1)
            n_ori_queries = grid_pts.shape[1]

            model_outputs = self.model(video=video, queries=grid_pts, iters=self.n_iters)

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

            traj_e[:, :, :, 0] *= (W - 1) / float(self.interp_shape[1] - 1)
            traj_e[:, :, :, 1] *= (H - 1) / float(self.interp_shape[0] - 1)

            traj_3d = torch.cat([traj_e, traj_d_e], dim=-1)

            traj_3d = traj_3d[:, :, :n_ori_queries]
            vis_e = vis_e[:, :, :n_ori_queries]

            dense_tracks[:, :, :, valid_y, valid_x] = rearrange(traj_3d, "b t n d -> b t d n")
            dense_visibilities[:, :, valid_y, valid_x] = vis_e

        # sparse_traj_e = rearrange(dense_traj_e[:, :, :, ::4, ::4], 'b t c h w -> b t (h w) c')
        # dense_traj_e = F.interpolate

        sparse_traj_e = rearrange(dense_tracks[:, :, :2, ::4, ::4], "b t c h w -> b t (h w) c")

        dense_tracks = rearrange(dense_tracks, "b t c h w -> b t (h w) c")
        dense_visibilities = rearrange(dense_visibilities, "b t h w -> b t (h w)")
        dense_visibilities = dense_visibilities > 0.8

        # dense_traj_e, dense_vis_e = self.interpolate_3d(dense_traj_e, dense_vis_e, ori_video, ori_videodepth)

        # # breakpoint()
        # dense_traj_3d = torch.cat([dense_traj_e, dense_traj_d_e], dim=-1)
        dense_trajs_3d = self.get_3d_trajs(dense_tracks, dense_visibilities, ori_video)

        # dense_down_traj_e = rearrange(dense_down_traj_e, 'b t c h w -> b t (h w) c')
        # dense_down_traj_d_e = rearrange(dense_down_traj_d_e, 'b t c h w -> b t (h w) c')
        # dense_down_vis_e = rearrange(dense_down_vis_e, 'b t h w -> b t (h w)')
        # dense_down_vis_e = (dense_down_vis_e > 0.8)

        # dense_traj_3d = torch.cat([dense_down_traj_e, dense_down_traj_d_e], dim=-1)
        # dense_trajs_3d = self.get_3d_trajs(dense_traj_3d, dense_down_vis_e, ori_video)

        out = {
            "sparse_tracks": sparse_traj_e,
            "tracks": dense_tracks[..., :2],
            "tracks_d": dense_tracks[..., 2:3],
            "vis": dense_visibilities,
            "trajs_3d": dense_trajs_3d,
        }

        # if return_virtual_tokens:
        #     out["virtual_tokens"] = cached_virtual_tokens
        return out

    def interpolate_3d(self, tracks, vis, ori_video, ori_videodepth):
        B, T, N, _ = tracks.shape
        H, W = ori_video.shape[3:]

        intr = torch.tensor(
            [
                [W, 0.0, W // 2],
                [0.0, H, H // 2],
                [0.0, 0.0, 1.0],
            ]
        ).to(tracks)

        grid_xy = get_grid(H, W, device=ori_video.device, normalize=False)

        grid_xy = grid_xy.view(1, H * W, 2).expand(B, -1, -1)  # B HW 2

        src_ = torch.cat(
            [
                tracks[:, 0, :, :2],  # B x N x 3,
                torch.ones_like(tracks[:, 0, :, 0:1]),
            ],
            dim=-1,
        )

        # src_[..., 0] *= (W - 1)
        # src_[..., 1] *= (H - 1)
        # src_[..., 2] = 1.0

        src_points_3d = torch.linalg.inv(intr[None, ...]) @ src_.reshape(-1, 3, 1)  # (TN) 3 1
        src_points_3d = src_points_3d.reshape(B, N, 3)  # B N 3
        src_points_3d *= tracks[:, 0, :, 2:3]

        # breakpoint()
        # src_depth_downsample = F.interpolate(src_depth, (H, W), mode='nearest').squeeze(1)
        grid_ = torch.cat(
            [
                grid_xy,
                torch.ones_like(grid_xy[..., 0:1]),
            ],
            dim=-1,
        )  # B x N x 3

        # grid_[..., 0] *= (W - 1)
        # grid_[..., 1] *= (H - 1)
        # grid_[..., 2] = 1.0

        grid_3d = torch.linalg.inv(intr[None, ...]) @ grid_.reshape(-1, 3, 1)  # (TN) 3 1
        grid_3d = grid_3d.reshape(B, -1, 3)  # T N 3
        grid_3d *= ori_videodepth[:, 0].reshape(B, -1, 1)

        # For each point in a regular grid, find indices of nearest visible source point
        src_pos, src_alpha = tracks[:, 0, :, :2], vis[:, 0, :]

        delta_tracks = tracks - tracks[:, 0:1]  # B T N 3
        delta_tracks = rearrange(delta_tracks, "b t n c -> b n (t c)")
        vis = rearrange(vis, "b t n -> b n t")

        # src_pos_packed = src_pos[src_alpha.bool()]
        # src_points_d_packed = src_points_d[src_alpha.bool()]
        # src_points_3d_packed = src_points_3d[src_alpha.bool()]

        # tgt_points_packed = tgt_points[src_alpha.bool()]
        # tgt_points_d_packed = tgt_points_d[src_alpha.bool()]

        lengths = src_alpha.sum(dim=1).long()
        max_length = int(lengths.max())
        cum_lengths = lengths.cumsum(dim=0)
        cum_lengths = torch.cat([torch.zeros_like(cum_lengths[:1]), cum_lengths[:-1]])

        # src_pos = pt3dops.packed_to_padded(src_pos_packed, cum_lengths, max_length)
        # src_points_d = pt3dops.packed_to_padded(src_points_d_packed, cum_lengths, max_length)
        src_points_3d = pt3dops.packed_to_padded(src_points_3d[src_alpha.bool()], cum_lengths, max_length)

        delta_tracks = pt3dops.packed_to_padded(delta_tracks[src_alpha.bool()], cum_lengths, max_length)
        vis = pt3dops.packed_to_padded(vis.float()[src_alpha.bool()], cum_lengths, max_length)

        # tgt_points = pt3dops.packed_to_padded(tgt_points_packed, cum_lengths, max_length)
        # tgt_points_d = pt3dops.packed_to_padded(tgt_points_d_packed, cum_lengths, max_length)

        _, idx, _ = pt3dops.knn_points(grid_3d, src_points_3d, lengths2=lengths, return_nn=False)
        idx = idx.view(B, H * W, 1)

        # breakpoint()
        dense_delta_tracks = pt3dops.knn_gather(delta_tracks, idx)  # b (h w) 1 (s c)
        dense_delta_tracks = rearrange(dense_delta_tracks, "b n 1 (t c) -> b t n c", t=T)

        dense_vis = pt3dops.knn_gather(vis, idx)  # b (h w) 1 (s c)
        dense_vis = rearrange(dense_vis, "b n 1 t -> b t n", t=T)

        # breakpoint()
        dense_tracks_finals = (
            torch.cat([grid_xy, ori_videodepth[:, 0].reshape(B, -1, 1)], dim=-1).unsqueeze(1) + dense_delta_tracks
        )

        dense_vis = dense_vis.bool()
        return dense_tracks_finals, dense_vis

    def get_3d_trajs(self, tracks, vis, video):
        video2d = video[0]  # T C H W
        H1, W1 = video[0].shape[-2:]

        xyzt = tracks[0]
        T = xyzt.shape[0]

        # H1, W1 = video[0].shape[-2:]
        intr = torch.tensor(
            [
                [W1, 0.0, W1 // 2],
                [0.0, W1, H1 // 2],
                [0.0, 0.0, 1.0],
            ]
        ).to(tracks)
        xyztVis = xyzt.clone()
        # xyztVis[..., 0] *= (ori_W - 1)
        # xyztVis[..., 1] *= (ori_H - 1)
        xyztVis[..., 2] = 1.0

        # breakpoint()

        # breakpoint()
        xyztVis = torch.linalg.inv(intr[None, ...]) @ xyztVis.reshape(-1, 3, 1)  # (TN) 3 1
        xyztVis = xyztVis.reshape(T, -1, 3)  # T N 3
        xyztVis[..., 2] *= xyzt[..., 2]

        pred_tracks2d = tracks[0][:, :, :2]
        # S1, N1, _ = pred_tracks2d.shape

        pred_tracks2dNm = pred_tracks2d.clone()
        pred_tracks2dNm[..., 0] = 2 * (pred_tracks2dNm[..., 0] / W1 - 0.5)
        pred_tracks2dNm[..., 1] = 2 * (pred_tracks2dNm[..., 1] / H1 - 0.5)
        # pred_tracks2dNm[..., 0] = 2*(pred_tracks2dNm[..., 0] - 0.5)
        # pred_tracks2dNm[..., 1] = 2*(pred_tracks2dNm[..., 1] - 0.5)
        color_interp = F.grid_sample(video2d, pred_tracks2dNm[:, :, None, :], align_corners=True)
        color_interp = color_interp[:, :, :, 0].permute(0, 2, 1)

        visible_pts = vis[0]

        colored_pts = torch.cat([xyztVis, color_interp, visible_pts.unsqueeze(-1)], dim=-1)

        return colored_pts
