# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from densetrack3d.models.model_utils import bilinear_sampler, get_points_on_a_grid
from einops import einsum, rearrange, repeat


class DensePredictor3D(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.interp_shape = model.model_resolution
        self.model = model
        self.n_iters = 6

    @torch.inference_mode()
    def forward(
        self,
        video,  # (B, T, 3, H, W)
        videodepth,
        grid_query_frame: int = 0,  # only for dense and regular grid tracks
        scale_input: bool = True,
        scale_to_origin: bool = True,
        use_efficient_global_attn: bool = True,
        predefined_intrs: torch.Tensor = None,
    ):
        B, T, C, H, W = video.shape
        device = video.device
        src_step = grid_query_frame

        ori_video = video.clone()

        if scale_input:
            video = F.interpolate(
                video.flatten(0, 1), tuple(self.interp_shape), mode="bilinear", align_corners=True
            ).reshape(B, T, 3, self.interp_shape[0], self.interp_shape[1])
            videodepth = F.interpolate(videodepth.flatten(0, 1), tuple(self.interp_shape), mode="nearest").reshape(
                B, T, 1, self.interp_shape[0], self.interp_shape[1]
            )

        if use_efficient_global_attn:
            sparse_xy = get_points_on_a_grid((36, 48), video.shape[3:]).long().float()
            sparse_xy = torch.cat([src_step * torch.ones_like(sparse_xy[:, :, :1]), sparse_xy], dim=2).to(
                device
            )  # B, N, C
            sparse_d = bilinear_sampler(
                videodepth[:, src_step], rearrange(sparse_xy[..., 1:3], "b n c -> b () n c"), mode="nearest"
            )
            sparse_d = rearrange(sparse_d, "b c m n -> b (m n) c")
            sparse_queries = torch.cat([sparse_xy, sparse_d], dim=2)  #
        else:
            sparse_queries = None

        sparse_predictions, dense_predictions, _ = self.model(
            video=video[:, src_step:],
            videodepth=videodepth[:, src_step:],
            sparse_queries=sparse_queries,
            iters=self.n_iters,
            use_efficient_global_attn=use_efficient_global_attn,
        )

        dense_traj_e, dense_traj_d_e, dense_vis_e = (
            dense_predictions["coords"],
            dense_predictions["coord_depths"],
            dense_predictions["vis"],
        )

        if "conf" not in dense_predictions.keys():
            dense_conf_e = dense_vis_e.clone()
        else:
            dense_conf_e = dense_predictions["conf"]

        if scale_to_origin:
            dense_traj_e[:, :, 0] *= (W - 1) / float(self.interp_shape[1] - 1)
            dense_traj_e[:, :, 1] *= (H - 1) / float(self.interp_shape[0] - 1)

        # sparse_traj_e = rearrange(dense_traj_e[:, :, :, ::4, ::4], 'b t c h w -> b t (h w) c')
        # sparse_vis_e = rearrange(dense_vis_e[:, :, ::4, ::4], 'b t h w -> b t (h w)')

        dense_traj_e = rearrange(dense_traj_e, "b t c h w -> b t (h w) c")
        dense_traj_d_e = rearrange(dense_traj_d_e, "b t c h w -> b t (h w) c")
        dense_vis_e = rearrange(dense_vis_e, "b t h w -> b t (h w)")
        dense_conf_e = rearrange(dense_conf_e, "b t h w -> b t (h w)")

        dense_vis_e = dense_vis_e > 0.8

        if scale_to_origin:
            dense_trajs_3d_dict = self.get_dense_trajs_3d(
                dense_traj_e,
                dense_traj_d_e,
                dense_vis_e,
                ori_video,
                query_frame=grid_query_frame,
                intr=predefined_intrs,
            )
        else:
            dense_trajs_3d_dict = self.get_dense_trajs_3d(
                dense_traj_e, dense_traj_d_e, dense_vis_e, video, query_frame=grid_query_frame, intr=predefined_intrs
            )

        out = {
            "trajs_uv": dense_traj_e,
            "trajs_depth": dense_traj_d_e,
            "trajs_3d_dict": dense_trajs_3d_dict,
            "vis": dense_vis_e,
            "conf": dense_conf_e,
            "dense_reso": self.interp_shape,
        }

        return out

    def get_dense_trajs_3d(self, trajs_uv, trajs_depth, vis, video, intr=None, query_frame=0):
        device = trajs_uv.device
        B, T, _, H, W = video.shape

        if intr is None:
            intr = torch.tensor(
                [
                    [W, 0.0, W // 2],
                    [0.0, W, H // 2],
                    [0.0, 0.0, 1.0],
                ]
            ).to(device)

        trajs_uv_homo = torch.cat([trajs_uv, torch.ones_like(trajs_uv[..., 0:1])], dim=-1)  # B T N 3

        xyz = einsum(trajs_uv_homo, torch.linalg.inv(intr), "b t n j, i j -> b t n i")
        xyz = xyz * trajs_depth

        query_rgb = video[:, query_frame]  # B 3 H W

        pred_tracks2dNm = trajs_uv[:, 0].clone()  #  B N 2
        pred_tracks2dNm[..., 0] = 2 * (pred_tracks2dNm[..., 0] / W - 0.5)
        pred_tracks2dNm[..., 1] = 2 * (pred_tracks2dNm[..., 1] / H - 0.5)
        color_interp = F.grid_sample(query_rgb, pred_tracks2dNm[:, :, None, :], align_corners=True)
        color_interp = rearrange(color_interp, "b c n 1 -> b n c")

        trajs_3d_dict = {
            "coords": xyz,
            "colors": color_interp,
            "vis": vis,
        }
        return trajs_3d_dict
