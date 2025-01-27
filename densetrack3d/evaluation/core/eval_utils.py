# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterable, Mapping, Tuple, Union

import numpy as np
import torch
from densetrack3d.models.model_utils import reduce_masked_mean, reduce_masked_median


def compute_lsfodyssey_metrics(
    rgbs, trajs_uv, trajs_z, trajs_vis, valids, trajs_uv_e, trajs_z_e, trajs_vis_e, intris, scaling_mode="median"
):
    # trajs_uv_e[:, :, :, 0] *= W / float(interp_shape[1])
    # trajs_uv_e[:, :, :, 1] *= H / float(interp_shape[0])  # shape: (B, T, N, 2)

    B, T, N = trajs_uv.shape[:3]
    H, W = rgbs.shape[-2:]

    # # NOTE scaling here
    # gt_depth_norm = torch.sqrt(torch.maximum(torch.tensor(1e-12, dtype=float, device=trajs_z.device), torch.square(trajs_z)))
    # pred_depth_norm = torch.sqrt(torch.maximum(torch.tensor(1e-12, dtype=float, device=trajs_z.device), torch.square(trajs_z_e)))
    # nan_mat = torch.full(gt_depth_norm.shape, torch.nan, device=gt_depth_norm.device)
    # valids_median_mask = trajs_vis.bool() & trajs_vis_e.bool() & valids.bool()
    # gt_depth_norm = torch.where(valids_median_mask, gt_depth_norm, nan_mat)
    # pred_depth_norm = torch.where(valids_median_mask, pred_depth_norm, nan_mat)

    # # breakpoint()
    # scale_factor = torch.nanmedian(gt_depth_norm.reshape(B,-1), dim=1)[0] / torch.nanmedian(pred_depth_norm.reshape(B,-1), dim=1)[0]
    # # breakpoint()
    # trajs_z_e = trajs_z_e * scale_factor
    #############################!SECTION

    # breakpoint()
    fx, fy, cx, cy = intris[0, 0, 0, 0], intris[0, 0, 1, 1], intris[0, 0, 0, 2], intris[0, 0, 1, 2]

    trajs_xyz_e = torch.zeros((B, T, N, 3), device=trajs_uv.device)
    trajs_xyz_e[:, :, :, 0] = trajs_z_e * (trajs_uv_e[:, :, :, 0] - cx) / fx
    trajs_xyz_e[:, :, :, 1] = trajs_z_e * (trajs_uv_e[:, :, :, 1] - cy) / fy
    trajs_xyz_e[:, :, :, 2] = trajs_z_e

    # breakpoint()

    trajs_xyz = torch.zeros((B, T, N, 3), device=trajs_uv.device)
    trajs_xyz[:, :, :, 0] = trajs_z * (trajs_uv[:, :, :, 0] - cx) / fx
    trajs_xyz[:, :, :, 1] = trajs_z * (trajs_uv[:, :, :, 1] - cy) / fy
    trajs_xyz[:, :, :, 2] = trajs_z

    # NOTE scaling here
    if scaling_mode == "median":
        gt_depth_norm = torch.sqrt(
            torch.maximum(
                torch.tensor(1e-12, dtype=float, device=trajs_z.device), torch.sum(torch.square(trajs_xyz), dim=-1)
            )
        )
        pred_depth_norm = torch.sqrt(
            torch.maximum(
                torch.tensor(1e-12, dtype=float, device=trajs_z.device), torch.sum(torch.square(trajs_xyz_e), dim=-1)
            )
        )
        nan_mat = torch.full(gt_depth_norm.shape, torch.nan, device=gt_depth_norm.device)
        valids_median_mask = trajs_vis.bool() & trajs_vis_e.bool() & valids.bool()
        gt_depth_norm = torch.where(valids_median_mask, gt_depth_norm, nan_mat)
        pred_depth_norm = torch.where(valids_median_mask, pred_depth_norm, nan_mat)

        # breakpoint()
        scale_factor = (
            torch.nanmedian(gt_depth_norm.reshape(B, -1), dim=1)[0]
            / torch.nanmedian(pred_depth_norm.reshape(B, -1), dim=1)[0]
        )
        # breakpoint()
        trajs_xyz_e = trajs_xyz_e * scale_factor

    res = torch.norm(trajs_xyz_e[:, 1:] - trajs_xyz[:, 1:], dim=-1)
    epe3d = torch.mean(res).item()
    acc3d_010 = reduce_masked_mean((res < 0.10).float(), valids[:, 1:]).item() * 100.0
    acc3d_020 = reduce_masked_mean((res < 0.20).float(), valids[:, 1:]).item() * 100.0
    acc3d_040 = reduce_masked_mean((res < 0.40).float(), valids[:, 1:]).item() * 100.0
    acc3d_080 = reduce_masked_mean((res < 0.80).float(), valids[:, 1:]).item() * 100.0
    acc3d_8 = (acc3d_080 + acc3d_010 + acc3d_020 + acc3d_040) / 4

    sur_thr = 0.50
    dists = torch.norm(trajs_xyz_e - trajs_xyz, dim=-1)  # B,S,N
    dist_ok = 1 - (dists > sur_thr).float() * valids.squeeze(-1)  # B,S,N
    survival_3d_5 = torch.cumprod(dist_ok, dim=1)  # B,S,N
    survival_3d_5 = torch.mean(survival_3d_5).item() * 100.0

    dists_ = dists.permute(0, 2, 1).reshape(B * N, T)
    valids_ = valids.permute(0, 2, 1).reshape(B * N, T)
    median_3d = reduce_masked_median(dists_, valids_, keep_batch=True)
    median_3d = median_3d.mean().item()

    sx_ = W / 256.0
    sy_ = H / 256.0
    sc_py = np.array([sx_, sy_]).reshape([1, 1, 2])
    sc_pt = torch.from_numpy(sc_py).float().cuda()

    thrs = [1, 2, 4, 8, 16]
    d_sum = 0.0
    for thr in thrs:
        # note we exclude timestep0 from this eval
        d_ = (torch.norm(trajs_uv_e[:, 1:] / sc_pt - trajs_uv[:, 1:] / sc_pt, dim=-1) < thr).float()  # B,S-1,N
        d_ = reduce_masked_mean(d_, valids[:, 1:]).item() * 100.0
        d_sum += d_
    d_avg = d_sum / len(thrs)

    sur_thr = 16
    dists = torch.norm(trajs_uv_e / sc_pt - trajs_uv / sc_pt, dim=-1)  # B,S,N
    dist_ok = 1 - (dists > sur_thr).float() * valids.squeeze(-1)  # B,S,N
    survival = torch.cumprod(dist_ok, dim=1)  # B,S,N
    survival = torch.mean(survival).item() * 100.0

    # get the median l2 error for each trajectory
    dists_ = dists.permute(0, 2, 1).reshape(B * N, T)
    valids_ = valids.permute(0, 2, 1).reshape(B * N, T)
    median_l2 = reduce_masked_median(dists_, valids_, keep_batch=True)
    median_l2 = median_l2.mean().item()

    metrics_tmp = {}
    metrics_tmp["acc3d_010"] = acc3d_010
    metrics_tmp["acc3d_020"] = acc3d_020
    metrics_tmp["acc3d_040"] = acc3d_040
    metrics_tmp["acc3d_080"] = acc3d_080
    metrics_tmp["acc3d_8"] = acc3d_8
    metrics_tmp["epe3d"] = epe3d
    metrics_tmp["survival_3d_5"] = survival_3d_5
    metrics_tmp["median_3d"] = median_3d
    metrics_tmp["d_avg"] = d_avg
    metrics_tmp["survival"] = survival
    metrics_tmp["median_l2"] = median_l2
    # count_all += 1

    return metrics_tmp


def compute_tapvid_metrics(
    query_points: np.ndarray,
    gt_occluded: np.ndarray,
    gt_tracks: np.ndarray,
    pred_occluded: np.ndarray,
    pred_tracks: np.ndarray,
    query_mode: str,
) -> Mapping[str, np.ndarray]:
    """Computes TAP-Vid metrics (Jaccard, Pts. Within Thresh, Occ. Acc.)
    See the TAP-Vid paper for details on the metric computation.  All inputs are
    given in raster coordinates.  The first three arguments should be the direct
    outputs of the reader: the 'query_points', 'occluded', and 'target_points'.
    The paper metrics assume these are scaled relative to 256x256 images.
    pred_occluded and pred_tracks are your algorithm's predictions.
    This function takes a batch of inputs, and computes metrics separately for
    each video.  The metrics for the full benchmark are a simple mean of the
    metrics across the full set of videos.  These numbers are between 0 and 1,
    but the paper multiplies them by 100 to ease reading.
    Args:
       query_points: The query points, an in the format [t, y, x].  Its size is
         [b, n, 3], where b is the batch size and n is the number of queries
       gt_occluded: A boolean array of shape [b, n, t], where t is the number
         of frames.  True indicates that the point is occluded.
       gt_tracks: The target points, of shape [b, n, t, 2].  Each point is
         in the format [x, y]
       pred_occluded: A boolean array of predicted occlusions, in the same
         format as gt_occluded.
       pred_tracks: An array of track predictions from your algorithm, in the
         same format as gt_tracks.
       query_mode: Either 'first' or 'strided', depending on how queries are
         sampled.  If 'first', we assume the prior knowledge that all points
         before the query point are occluded, and these are removed from the
         evaluation.
    Returns:
        A dict with the following keys:
        occlusion_accuracy: Accuracy at predicting occlusion.
        pts_within_{x} for x in [1, 2, 4, 8, 16]: Fraction of points
          predicted to be within the given pixel threshold, ignoring occlusion
          prediction.
        jaccard_{x} for x in [1, 2, 4, 8, 16]: Jaccard metric for the given
          threshold
        average_pts_within_thresh: average across pts_within_{x}
        average_jaccard: average across jaccard_{x}
    """

    metrics = {}
    # Fixed bug is described in:
    # https://github.com/facebookresearch/co-tracker/issues/20
    eye = np.eye(gt_tracks.shape[2], dtype=np.int32)

    if query_mode == "first":
        # evaluate frames after the query frame
        query_frame_to_eval_frames = np.cumsum(eye, axis=1) - eye
    elif query_mode == "strided":
        # evaluate all frames except the query frame
        query_frame_to_eval_frames = 1 - eye
    else:
        raise ValueError("Unknown query mode " + query_mode)

    query_frame = query_points[..., 0]
    query_frame = np.round(query_frame).astype(np.int32)
    evaluation_points = query_frame_to_eval_frames[query_frame] > 0

    # Occlusion accuracy is simply how often the predicted occlusion equals the
    # ground truth.
    occ_acc = np.sum(
        np.equal(pred_occluded, gt_occluded) & evaluation_points,
        axis=(1, 2),
    ) / np.sum(evaluation_points)
    metrics["occlusion_accuracy"] = occ_acc

    # Next, convert the predictions and ground truth positions into pixel
    # coordinates.
    visible = np.logical_not(gt_occluded)
    pred_visible = np.logical_not(pred_occluded)
    all_frac_within = []
    all_jaccard = []
    for thresh in [1, 2, 4, 8, 16]:
        # True positives are points that are within the threshold and where both
        # the prediction and the ground truth are listed as visible.
        within_dist = np.sum(
            np.square(pred_tracks - gt_tracks),
            axis=-1,
        ) < np.square(thresh)
        is_correct = np.logical_and(within_dist, visible)

        # Compute the frac_within_threshold, which is the fraction of points
        # within the threshold among points that are visible in the ground truth,
        # ignoring whether they're predicted to be visible.
        count_correct = np.sum(
            is_correct & evaluation_points,
            axis=(1, 2),
        )
        count_visible_points = np.sum(visible & evaluation_points, axis=(1, 2))
        frac_correct = count_correct / count_visible_points
        metrics["pts_within_" + str(thresh)] = frac_correct
        all_frac_within.append(frac_correct)

        true_positives = np.sum(is_correct & pred_visible & evaluation_points, axis=(1, 2))

        # The denominator of the jaccard metric is the true positives plus
        # false positives plus false negatives.  However, note that true positives
        # plus false negatives is simply the number of points in the ground truth
        # which is easier to compute than trying to compute all three quantities.
        # Thus we just add the number of points in the ground truth to the number
        # of false positives.
        #
        # False positives are simply points that are predicted to be visible,
        # but the ground truth is not visible or too far from the prediction.
        gt_positives = np.sum(visible & evaluation_points, axis=(1, 2))
        false_positives = (~visible) & pred_visible
        false_positives = false_positives | ((~within_dist) & pred_visible)
        false_positives = np.sum(false_positives & evaluation_points, axis=(1, 2))
        jaccard = true_positives / (gt_positives + false_positives)
        metrics["jaccard_" + str(thresh)] = jaccard
        all_jaccard.append(jaccard)
    metrics["average_jaccard"] = np.mean(
        np.stack(all_jaccard, axis=1),
        axis=1,
    )
    metrics["average_pts_within_thresh"] = np.mean(
        np.stack(all_frac_within, axis=1),
        axis=1,
    )
    return metrics
