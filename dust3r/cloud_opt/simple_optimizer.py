# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Main class for the implementation of the global alignment
# --------------------------------------------------------
from tqdm import tqdm
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum

from dust3r.cloud_opt.pair_viewer import PairViewer
from dust3r.cloud_opt.base_opt import BasePCOptimizer
from dust3r.utils.geometry import xy_grid, geotrf
from dust3r.utils.device import to_cpu, to_numpy
from dust3r.utils.geometry_opt import DepthBasedWarping, WarpImage, OccMask, DepthWarpWithFlow, depth_regularization_si_weighted

def smooth_L1_loss_fn(estimate, gt, mask, beta=1.0, per_pixel_thre=50.):
    loss_raw_shape = F.smooth_l1_loss(estimate*mask, gt*mask, beta=beta, reduction='none')
    if per_pixel_thre > 0:
        per_pixel_mask = (loss_raw_shape < per_pixel_thre) * mask
    else:
        per_pixel_mask = mask
    return torch.sum(loss_raw_shape * per_pixel_mask) / torch.sum(per_pixel_mask)

def mse_loss_fn(estimate, gt, mask):
    v = torch.sum((estimate*mask-gt*mask)**2) / torch.sum(mask)
    return v  # , v.item()

class SimplePointCloudOptimizer(BasePCOptimizer):
    """ Optimize a global scene, given a list of pairwise observations.
    Graph node: images
    Graph edges: observations = (pred1, pred2)
    """

    def __init__(self, 
                *args, 
                optimize_pp=False, 
                focal_break=20, 
                init_depth=None, 
                disable_flow_loss=None, 
                shared_focal=True,
                motion_mask_thre=0.3,
                num_total_iter=300,
                pxl_thre=50,
                flow_loss_fn='smooth_l1',
                is_static=False,
                use_self_dynamic_mask=False,
                **kwargs):
        super().__init__(*args, **kwargs)

        self.has_im_poses = True  # by definition of this class
        self.focal_break = focal_break

        self.num_total_iter = num_total_iter
        self.motion_mask_thre = motion_mask_thre
        self.pxl_thre = pxl_thre
        self.is_static = is_static
        self.use_self_dynamic_mask = use_self_dynamic_mask

        # adding thing to optimize
        self.im_depthmaps = nn.ParameterList(torch.randn(H, W)/10-3 for H, W in self.imshapes)  # log(depth)
        self.im_poses = nn.ParameterList(self.rand_pose(self.POSE_DIM) for _ in range(self.n_imgs))  # camera poses
        # self.im_focals = nn.ParameterList(torch.FloatTensor(
        #     [self.focal_break*np.log(max(H, W))]) for H, W in self.imshapes)  # camera intrinsics
        self.shared_focal = shared_focal
        if self.shared_focal:
            self.im_focals = nn.ParameterList(torch.FloatTensor(
                [self.focal_break*np.log(max(H, W))]) for H, W in self.imshapes[:1])  # camera intrinsics
        else:
            self.im_focals = nn.ParameterList(torch.FloatTensor(
                [self.focal_break*np.log(max(H, W))]) for H, W in self.imshapes)  # camera intrinsics
        
        
        self.im_pp = nn.ParameterList(torch.zeros((2,)) for _ in range(self.n_imgs))  # camera intrinsics
        self.im_pp.requires_grad_(optimize_pp)

        self.imshape = self.imshapes[0]
        im_areas = [h*w for h, w in self.imshapes]
        self.max_area = max(im_areas)

        # adding thing to optimize
        self.im_depthmaps = ParameterStack(self.im_depthmaps, is_param=True, fill=self.max_area)
        self.im_poses = ParameterStack(self.im_poses, is_param=True)
        self.im_focals = ParameterStack(self.im_focals, is_param=True)
        self.im_pp = ParameterStack(self.im_pp, is_param=True)
        self.register_buffer('_pp', torch.tensor([(w/2, h/2) for h, w in self.imshapes]))
        self.register_buffer('_grid', ParameterStack(
            [xy_grid(W, H, device=self.device) for H, W in self.imshapes], fill=self.max_area))

        # pre-compute pixel weights
        self.register_buffer('_weight_i', ParameterStack(
            [self.conf_trf(self.conf_i[i_j]) for i_j in self.str_edges], fill=self.max_area))
        self.register_buffer('_weight_j', ParameterStack(
            [self.conf_trf(self.conf_j[i_j]) for i_j in self.str_edges], fill=self.max_area))

        # precompute aa
        self.register_buffer('_stacked_pred_i', ParameterStack(self.pred_i, self.str_edges, fill=self.max_area))
        self.register_buffer('_stacked_pred_j', ParameterStack(self.pred_j, self.str_edges, fill=self.max_area))
        self.register_buffer('_ei', torch.tensor([i for i, j in self.edges]))
        self.register_buffer('_ej', torch.tensor([j for i, j in self.edges]))
        self.total_area_i = sum([im_areas[i] for i, j in self.edges])
        self.total_area_j = sum([im_areas[j] for i, j in self.edges])


        if flow_loss_fn == 'smooth_l1':
            self.flow_loss_fn = smooth_L1_loss_fn
        elif flow_loss_fn == 'mse':
            self.flow_loss_fn = mse_loss_fn

        self.depth_wrapper = DepthBasedWarping()
        self.backward_warper = WarpImage()

        self.flow_ij, self.flow_ji, self.flow_valid_mask_i, self.flow_valid_mask_j = self.get_flow() # (num_pairs, 2, H, W)

        if hasattr(self, 'visib_i') and hasattr(self, 'visib_j'):
            self.visib_ij, self.visib_ji = self.get_visib(self.flow_valid_mask_i, self.flow_valid_mask_j)
        
        if self.is_static:
            print("Static scene, set dynamic masks to zero")
            self.dynamic_masks = [torch.zeros(self.imshape[0], self.imshape[1], dtype=torch.bool, device=self.device) for _ in range(self.n_imgs)]
        else:
            if self.dynamic_masks is None or self.use_self_dynamic_mask:
                self.get_motion_mask_from_pairs(*args)
        self.depth_warper_with_flow = DepthWarpWithFlow()

        # breakpoint()
        # FIXME NO pw_poses for simple optimizer
        self.pw_poses.requires_grad_(False)
        self.pw_adaptors.requires_grad_(False)

        self.loss_weight_dict = {
            'align_loss': 1.0,
            'smooth_loss': 0.01,
            'flow_loss': 0.005,
            'consistent_depth_loss': 0.5,
        }

        if disable_flow_loss:
            self.loss_weight_dict['flow_loss'] = 0.0

        if init_depth is not None:
            self.init_depth = init_depth
            self.set_depth = True
            for k, v in self.init_depth.items():
                self._set_depthmap(k, v, force=True)
        else:
            self.set_depth = False

    def get_flow(self): #TODO: test with gt flow
        print('precomputing flow...')
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        get_valid_flow_mask = OccMask(th=3.0)
        # pair_imgs = [np.stack(self.imgs)[self._ei], np.stack(self.imgs)[self._ej]]
        flow_ij = torch.stack([self.flow2d_i[i_j] for i_j in self.str_edges], dim=0).permute(0,3,1,2) # T 2 H W
        flow_ji = torch.stack([self.flow2d_j[i_j] for i_j in self.str_edges], dim=0).permute(0,3,1,2) # T 2 H W
        valid_mask_i = get_valid_flow_mask(flow_ij, flow_ji)
        valid_mask_j = get_valid_flow_mask(flow_ji, flow_ij)
        print('flow precomputed')
        # delete the flow net
        # if flow_net is not None: del flow_net
        return flow_ij, flow_ji, valid_mask_i, valid_mask_j
    
    def get_visib(self, valid_mask_i=None, valid_mask_j=None): 
        print('precomputing visib')
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # get_valid_flow_mask = OccMask(th=3.0)
        # pair_imgs = [np.stack(self.imgs)[self._ei], np.stack(self.imgs)[self._ej]]
        visib_ij = torch.stack([self.visib_i[i_j] for i_j in self.str_edges], dim=0) # T H W
        visib_ji = torch.stack([self.visib_j[i_j] for i_j in self.str_edges], dim=0) # T H W
        
        if valid_mask_i is None:
            visib_ij = visib_ij * valid_mask_i
        if valid_mask_j is None:
            visib_ji = visib_ji * valid_mask_j

        return visib_ij, visib_ji

    def get_motion_mask_from_pairs(self, view1, view2, pred1, pred2):
        # assert self.is_symmetrized, 'only support symmetric case'

        print("Get dynamic masks from pairs")
        symmetry_pairs_idx = [(i, i+len(self.edges)//2) for i in range(len(self.edges)//2)]
        intrinsics_i = []
        intrinsics_j = []
        R_i = []
        R_j = []
        T_i = []
        T_j = []
        depth_maps_i = []
        depth_maps_j = []
        for i, j in tqdm(symmetry_pairs_idx):
            new_view1 = {}
            new_view2 = {}
            for key in view1.keys():
                if isinstance(view1[key], list):
                    new_view1[key] = [view1[key][i], view1[key][j]]
                    new_view2[key] = [view2[key][i], view2[key][j]]
                elif isinstance(view1[key], torch.Tensor):
                    new_view1[key] = torch.stack([view1[key][i], view1[key][j]]).cpu()
                    new_view2[key] = torch.stack([view2[key][i], view2[key][j]]).cpu()
            new_view1['idx'] = [0, 1]
            new_view2['idx'] = [1, 0]
            new_pred1 = {}
            new_pred2 = {}
            for key in pred1.keys():
                if key in ["flow3d", "flow2d"]:
                    continue
                if isinstance(pred1[key], list):
                    new_pred1[key] = [pred1[key][i], pred1[key][j]]
                elif isinstance(pred1[key], torch.Tensor):
                    new_pred1[key] = torch.stack([pred1[key][i], pred1[key][j]]).cpu()

            for key in pred2.keys():
                if key in ["flow3d", "flow2d"]:
                    continue
                if isinstance(pred2[key], list):
                    new_pred2[key] = [pred2[key][i], pred2[key][j]]
                elif isinstance(pred2[key], torch.Tensor):
                    new_pred2[key] = torch.stack([pred2[key][i], pred2[key][j]]).cpu()
            pair_viewer = PairViewer(new_view1, new_view2, new_pred1, new_pred2, verbose=False, min_conf_thr=1.0 + math.exp(0.9))
            intrinsics_i.append(pair_viewer.get_intrinsics()[0])
            intrinsics_j.append(pair_viewer.get_intrinsics()[1])
            R_i.append(pair_viewer.get_im_poses()[0][:3, :3])
            R_j.append(pair_viewer.get_im_poses()[1][:3, :3])
            T_i.append(pair_viewer.get_im_poses()[0][:3, 3:])
            T_j.append(pair_viewer.get_im_poses()[1][:3, 3:])
            depth_maps_i.append(pair_viewer.get_depthmaps()[0])
            depth_maps_j.append(pair_viewer.get_depthmaps()[1])
        
        self.intrinsics_i = torch.stack(intrinsics_i).to(self.flow_ij.device)
        self.intrinsics_j = torch.stack(intrinsics_j).to(self.flow_ij.device)
        self.R_i = torch.stack(R_i).to(self.flow_ij.device)
        self.R_j = torch.stack(R_j).to(self.flow_ij.device)
        self.T_i = torch.stack(T_i).to(self.flow_ij.device)
        self.T_j = torch.stack(T_j).to(self.flow_ij.device)
        self.depth_maps_i = torch.stack(depth_maps_i).unsqueeze(1).to(self.flow_ij.device)
        self.depth_maps_j = torch.stack(depth_maps_j).unsqueeze(1).to(self.flow_ij.device)

        ego_flow_1_2, _ = self.depth_wrapper(self.R_i, self.T_i, self.R_j, self.T_j, 1 / (self.depth_maps_i + 1e-6), self.intrinsics_j, torch.linalg.inv(self.intrinsics_i))
        ego_flow_2_1, _ = self.depth_wrapper(self.R_j, self.T_j, self.R_i, self.T_i, 1 / (self.depth_maps_j + 1e-6), self.intrinsics_i, torch.linalg.inv(self.intrinsics_j))
        # ego_flow_1_2, _ = self.depth_wrapper(self.R_i, self.T_i, self.R_j, self.T_j, self.depth_maps_i, self.intrinsics_j, torch.linalg.inv(self.intrinsics_i), use_depth=True)
        # ego_flow_2_1, _ = self.depth_wrapper(self.R_j, self.T_j, self.R_i, self.T_i, self.depth_maps_j, self.intrinsics_i, torch.linalg.inv(self.intrinsics_j), use_depth=True)

        # breakpoint()
        err_map_i = torch.norm(ego_flow_1_2[:, :2, ...] - self.flow_ij[:len(symmetry_pairs_idx)], dim=1)
        err_map_j = torch.norm(ego_flow_2_1[:, :2, ...] - self.flow_ji[:len(symmetry_pairs_idx)], dim=1)
        # normalize the error map for each pair
        err_map_i = (err_map_i - err_map_i.amin(dim=(1, 2), keepdim=True)) / (err_map_i.amax(dim=(1, 2), keepdim=True) - err_map_i.amin(dim=(1, 2), keepdim=True))
        err_map_j = (err_map_j - err_map_j.amin(dim=(1, 2), keepdim=True)) / (err_map_j.amax(dim=(1, 2), keepdim=True) - err_map_j.amin(dim=(1, 2), keepdim=True))
        self.dynamic_masks = [[] for _ in range(self.n_imgs)]

        for i, j in symmetry_pairs_idx:
            i_idx = self._ei[i]
            j_idx = self._ej[i]
            self.dynamic_masks[i_idx].append(err_map_i[i])
            self.dynamic_masks[j_idx].append(err_map_j[i])
        
        for i in range(self.n_imgs):
            self.dynamic_masks[i] = torch.stack(self.dynamic_masks[i]).mean(dim=0) > self.motion_mask_thre

    def _check_all_imgs_are_selected(self, msk):
        assert np.all(self._get_msk_indices(msk) == np.arange(self.n_imgs)), 'incomplete mask!'

    def preset_pose(self, known_poses, pose_msk=None):  # cam-to-world
        self._check_all_imgs_are_selected(pose_msk)

        if isinstance(known_poses, torch.Tensor) and known_poses.ndim == 2:
            known_poses = [known_poses]
        for idx, pose in zip(self._get_msk_indices(pose_msk), known_poses):
            if self.verbose:
                print(f' (setting pose #{idx} = {pose[:3,3]})')
            self._no_grad(self._set_pose(self.im_poses, idx, torch.tensor(pose)))

        # normalize scale if there's less than 1 known pose
        n_known_poses = sum((p.requires_grad is False) for p in self.im_poses)
        self.norm_pw_scale = (n_known_poses <= 1)

        self.im_poses.requires_grad_(False)
        self.norm_pw_scale = False

    def preset_focal(self, known_focals, msk=None):
        self._check_all_imgs_are_selected(msk)

        for idx, focal in zip(self._get_msk_indices(msk), known_focals):
            if self.verbose:
                print(f' (setting focal #{idx} = {focal})')
            self._no_grad(self._set_focal(idx, focal))

        self.im_focals.requires_grad_(False)

    def preset_principal_point(self, known_pp, msk=None):
        self._check_all_imgs_are_selected(msk)

        for idx, pp in zip(self._get_msk_indices(msk), known_pp):
            if self.verbose:
                print(f' (setting principal point #{idx} = {pp})')
            self._no_grad(self._set_principal_point(idx, pp))

        self.im_pp.requires_grad_(False)

    def freeze_pose(self):
        self.im_poses.requires_grad_(False)
        self.im_pp.requires_grad_(False)
        self.im_focals.requires_grad_(False)

    def _get_msk_indices(self, msk):
        if msk is None:
            return range(self.n_imgs)
        elif isinstance(msk, int):
            return [msk]
        elif isinstance(msk, (tuple, list)):
            return self._get_msk_indices(np.array(msk))
        elif msk.dtype in (bool, torch.bool, np.bool_):
            assert len(msk) == self.n_imgs
            return np.where(msk)[0]
        elif np.issubdtype(msk.dtype, np.integer):
            return msk
        else:
            raise ValueError(f'bad {msk=}')

    def _no_grad(self, tensor):
        assert tensor.requires_grad, 'it must be True at this point, otherwise no modification occurs'

    def _set_focal(self, idx, focal, force=False):
        param = self.im_focals[idx]
        if param.requires_grad or force:  # can only init a parameter not already initialized
            param.data[:] = self.focal_break * np.log(focal)
        return param

    def get_focals(self):
        if self.shared_focal:
            log_focals = torch.stack([self.im_focals[0]] * self.n_imgs, dim=0)
        else:
            log_focals = torch.stack(list(self.im_focals), dim=0)
        return (log_focals / self.focal_break).exp()

    def get_known_focal_mask(self):
        return torch.tensor([not (p.requires_grad) for p in self.im_focals])

    def _set_principal_point(self, idx, pp, force=False):
        param = self.im_pp[idx]
        H, W = self.imshapes[idx]
        if param.requires_grad or force:  # can only init a parameter not already initialized
            param.data[:] = to_cpu(to_numpy(pp) - (W/2, H/2)) / 10
        return param

    def get_principal_points(self):
        return self._pp + 10 * self.im_pp

    def get_intrinsics(self):
        K = torch.zeros((self.n_imgs, 3, 3), device=self.device)
        focals = self.get_focals().flatten()
        K[:, 0, 0] = K[:, 1, 1] = focals
        K[:, :2, 2] = self.get_principal_points()
        K[:, 2, 2] = 1
        return K

    def get_im_poses(self):  # cam to world
        cam2world = self._get_poses(self.im_poses)
        return cam2world

    def _set_depthmap(self, idx, depth, force=False):
        depth = _ravel_hw(depth, self.max_area)

        param = self.im_depthmaps[idx]
        if param.requires_grad or force:  # can only init a parameter not already initialized
            param.data[:] = depth.log().nan_to_num(neginf=0)
        return param

    # def freeze_depthmap(self, idx):
    #     param = self.im_depthmaps[idx]
    #     param = param.no_grad()

    def get_depthmaps(self, raw=False):
        res = self.im_depthmaps.exp()
        if not raw:
            res = [dm[:h*w].view(h, w) for dm, (h, w) in zip(res, self.imshapes)]
        return res

    def depth_to_pts3d(self, return_pose=False):
        # Get depths and  projection params if not provided
        focals = self.get_focals()
        pp = self.get_principal_points()
        im_poses = self.get_im_poses()
        depth = self.get_depthmaps(raw=True)

        # get pointmaps in camera frame
        rel_ptmaps = _fast_depthmap_to_pts3d(depth, self._grid, focals, pp=pp)
        # project to world frame
        if return_pose:
            return rel_ptmaps, im_poses
        return rel_ptmaps

    def get_pts3d(self, raw=False, return_pose=False, fuse=False):
        out = self.depth_to_pts3d(return_pose=return_pose)
        if return_pose:
            res, pose = out
        else:
            res = out

        if fuse:
            # assert not return_pose
            res = geotrf(pose, res)

        if not raw:
            res = [dm[:h*w].view(h, w, 3) for dm, (h, w) in zip(res, self.imshapes)]

        if return_pose:
            return res, pose
        return res

    def forward(self, cur_iter, stage='1'):
        # FIXME hardcode now:
        # if self.set_depth:
        #     for k, v in self.init_depth.items():
        #         self._set_depthmap(k, v, force=True)

        # breakpoint()
        # pw_poses = self.get_pw_poses()  # cam-to-world
        # pw_adapt = self.get_adaptors().unsqueeze(1)
        rel_ptmaps, im_poses = self.get_pts3d(raw=True, return_pose=True)

        proj_pts3d = geotrf(im_poses, rel_ptmaps)

        # if torch.any(torch.isnan(pw_poses)):
        #     print("nan pose")
        #     breakpoint()

        
        # if torch.any(torch.isnan(pw_adapt)):
        #     print("nan pw_adapt")
        #     breakpoint()

        
        # if torch.any(torch.isnan(proj_pts3d)):
        #     print("nan proj_pts3d")
        #     breakpoint()

        # rotate pairwise prediction according to pw_poses

        pose_ei = im_poses[self._ei]
        # pose_ej = im_poses[self._ej]
        aligned_pred_i = geotrf(pose_ei, self._stacked_pred_i)
        aligned_pred_j = geotrf(pose_ei, self._stacked_pred_j)

        # breakpoint()
        # compute the less
        li = self.dist(proj_pts3d[self._ei], aligned_pred_i, weight=self._weight_i).sum() / self.total_area_i
        lj = self.dist(proj_pts3d[self._ej], aligned_pred_j, weight=self._weight_j).sum() / self.total_area_j
        # breakpoint()

        align_loss = li + lj

        # NOTE enable smooth loss:
        # smooth_loss = self.get_smooth_loss(im_poses)
        # flow_loss = self.get_flow2d_loss(im_poses, rel_ptmaps)
        smooth_loss = self.get_smooth_loss_monst3r(im_poses)
        flow_loss = self.get_flow2d_loss_monst3r(cur_iter)
        # total_loss = align_loss + smooth_loss + flow_loss

        loss_dict = {
            'align_loss': align_loss * self.loss_weight_dict['align_loss'],
            'smooth_loss': smooth_loss * self.loss_weight_dict['smooth_loss'],
            'flow_loss': flow_loss * self.loss_weight_dict['flow_loss'],
        }

        # if stage == '2':
            # loss_dict = {}

        # NOTE consistent depth loss
        # consistent_depth_loss = self.get_consistent_depth_loss()
        # loss_dict['consistent_depth_loss'] = consistent_depth_loss * self.loss_weight_dict['consistent_depth_loss']

        # print("consist depth loss: ", loss_dict['consistent_depth_loss'].item())
        # breakpoint()
        return loss_dict
    
    def get_consistent_depth_loss(self):

        depthmaps = torch.stack(self.get_depthmaps(raw=False)).unsqueeze(1)
        dynamic_masks_all = torch.stack(self.dynamic_masks).to(self.device).unsqueeze(1)

        dynamic_mask_i = dynamic_masks_all[self._ei]
        depthmaps_i = depthmaps[self._ei].detach() # n_pair, H W
        depthmaps_j = depthmaps[self._ej] # n_pair, H W
        # grids_i = grids[self._ei]   # n_pair, 2 H W

        flow_ij = self.flow_ij
        visib_ij = self.visib_ij

        _, T_all = self.get_im_poses()[:,:3].split([3, 1], dim=-1)
        T_j, T_i = T_all[self._ej].squeeze(-1), T_all[self._ei].squeeze(-1)
        compensated_trans = torch.norm(T_j - T_i, dim=-1, p=2)

        # valid_pairs = (self._ei == 0) | (self._ei == 10) | (self._ei == 20) | (self._ei == 30)
        # dynamic_mask_i = dynamic_mask_i[valid_pairs]
        # depthmaps_i = depthmaps_i[valid_pairs]
        # depthmaps_j = depthmaps_j[valid_pairs]
        # compensated_trans = compensated_trans[valid_pairs]
        # flow_ij = flow_ij[valid_pairs]
        # visib_ij = visib_ij[valid_pairs]


        warped_depth, inbound_mask = self.depth_warper_with_flow(flow_ij, depthmaps_j)
        inbound_mask = inbound_mask.float()
        # warped_depth = warped_depth.squeeze(1)
        # depthmaps_i = depthmaps_i.squeeze(1)

        # breakpoint()
        valid_mask = visib_ij.unsqueeze(1) * inbound_mask * (1 - dynamic_mask_i)


        compensated_trans = compensated_trans[:, None, None, None] # n_pair, 1, 1, 1


        # breakpoint()
        warped_depth = warped_depth + compensated_trans

        # NOTE mask out 

        # breakpoint()
        consistent_depth_loss = F.l1_loss(warped_depth, depthmaps_i, reduction='none') * valid_mask
        consistent_depth_loss = consistent_depth_loss.sum() / valid_mask.sum()
        # warped_depth[~valid_mask.bool()] = 1
        # depthmaps_i[~depthmaps_i.bool()] = 1
        # consistent_depth_loss = depth_regularization_si_weighted(warped_depth, depthmaps_i)

        # breakpoint()

        return consistent_depth_loss
    
    def get_smooth_loss(self, im_poses):
        # im_poses = self.get_im_poses()
    
        R = im_poses[:, :3, :3]
        T = im_poses[:, :3, 3]

        smooth_t_loss = einsum(R[:-1].permute(0,2,1), (T[1:] - T[:-1])[..., None], 't n m, t m k -> t n k').squeeze(-1) # T, 3
        smooth_t_loss = smooth_t_loss.norm(dim=1, p=2)
        # breakpoint()
        smooth_t_loss = smooth_t_loss.mean()

        smooth_r_loss = einsum(R[:-1].permute(0,2,1), R[1:], 't n m, t m k -> t n k') # T 3 3
        smooth_r_loss = (smooth_r_loss - torch.eye(3, device=self.device)[None]).norm(dim=(1, 2), p='fro')
        # breakpoint()
        smooth_r_loss = smooth_r_loss.mean()

        smooth_loss = smooth_t_loss + smooth_r_loss
        # smooth_loss = 0.1 * smooth_loss

        return smooth_loss

    def get_smooth_loss_monst3r(self, im_poses):
        # im_poses = self.get_im_poses()
        RT1, RT2 = im_poses[:-1], im_poses[1:]

        relative_RT = torch.matmul(torch.inverse(RT1), RT2)
        rotation_diff = relative_RT[:, :3, :3]
        translation_diff = relative_RT[:, :3, 3]

        # Frobenius norm for rotation difference
        rotation_loss = torch.norm(rotation_diff - (torch.eye(3, device=RT1.device)), dim=(1, 2))

        # L2 norm for translation difference
        translation_loss = torch.norm(translation_diff, dim=1)

        # Combined loss (one can weigh these differently if needed)
        pose_loss = rotation_loss + translation_loss * 1.0
        pose_loss = pose_loss.sum()
        return pose_loss
    
    def get_flow_loss(self, im_poses, rel_ptmaps):
        if not hasattr(self, 'flow3d_i') or not hasattr(self, 'dynamic_masks'):
            return torch.zeros(0, device=self.device, requires_grad=True)
        
        flow3d_i_dict = self.flow3d_i
        flow3d_i_gt = torch.stack([flow3d_i_dict[i_j] for i_j in self.str_edges], dim=0)
        flow3d_i_gt = rearrange(flow3d_i_gt, 't h w c -> t (h w) c')

        # proj_pts3d = self.get_pts3d(raw=True) # in wolrd coordinates
        # im_poses = self.get_im_poses()
        
        proj_pts3d_at_i = rel_ptmaps[self._ei]

        im_poses_i = im_poses[self._ei] # cam i to world
        im_poses_j = im_poses[self._ej] # cam j to world

        im_poses_i_to_j = einsum(torch.linalg.inv(im_poses_j), im_poses_i, 't m n, t n k -> t m k') # cam i to cam j

        proj_pts3d_at_j = geotrf(im_poses_i_to_j, proj_pts3d_at_i) # in cam j coordinates

        pred_flow3d = proj_pts3d_at_j - proj_pts3d_at_i


        # breakpoint()
        flow_diff = pred_flow3d - flow3d_i_gt
        dynamic_masks = self.dynamic_masks[self._ei]
        dynamic_masks = rearrange(dynamic_masks, 't h w -> t (h w)')
        static_masks = 1 - dynamic_masks
        flow_loss = (flow_diff.abs().sum(-1) * static_masks).sum() / static_masks.sum()

        # breakpoint()
        # flow_loss = flow_loss * 0.01
        return flow_loss
    
    def get_flow2d_loss(self, im_poses, rel_ptmaps):
        if not hasattr(self, 'flow2d_i') or not hasattr(self, 'dynamic_masks'):
            return torch.zeros(0, device=self.device, requires_grad=True)
        
        focals = self.get_focals()
        pp = self.get_principal_points()
        
        flow2d_i_dict = self.flow2d_i
        flow2d_i_gt = torch.stack([flow2d_i_dict[i_j] for i_j in self.str_edges], dim=0)
        flow2d_i_gt = rearrange(flow2d_i_gt, 't h w c -> t (h w) c')

        proj_pts3d_at_i = rel_ptmaps[self._ei]
        pts2d_at_i = _fast_pts3d_to_2d(proj_pts3d_at_i, focals[self._ei], pp[self._ei])

        im_poses_i = im_poses[self._ei] # cam i to world
        im_poses_j = im_poses[self._ej] # cam j to world

        im_poses_i_to_j = einsum(torch.linalg.inv(im_poses_j), im_poses_i, 't m n, t n k -> t m k') # cam i to cam j
        proj_pts3d_at_j = geotrf(im_poses_i_to_j, proj_pts3d_at_i) # in cam j coordinates
        pts2d_at_j = _fast_pts3d_to_2d(proj_pts3d_at_j, focals[self._ej], pp[self._ej])

        pred_flow2d = pts2d_at_j - pts2d_at_i


        # breakpoint()
        flow_diff = pred_flow2d - flow2d_i_gt

        dynamic_masks_all = torch.stack(self.dynamic_masks)[self._ei].to(self.device).float()
        # dynamic_masks = self.dynamic_masks[self._ei]
        dynamic_masks_all = rearrange(dynamic_masks_all, 't h w -> t (h w)')
        static_masks_all = 1 - dynamic_masks_all
        flow_loss = (flow_diff.abs().sum(-1) * static_masks_all).sum() / static_masks_all.sum()

        # breakpoint()
        # flow_loss = flow_loss * 0.01
        return flow_loss

    def get_flow2d_loss_monst3r(self, cur_iter):
        if not hasattr(self, 'flow2d_i') or not hasattr(self, 'dynamic_masks'):
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        # torch.zeros(1, device=self.device, requires_grad=True)

        if cur_iter <= (0.1 * self.num_total_iter):
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        R_all, T_all = self.get_im_poses()[:,:3].split([3, 1], dim=-1)
        R1, T1 = R_all[self._ei], T_all[self._ei]
        R2, T2 = R_all[self._ej], T_all[self._ej]
        K_all = self.get_intrinsics()
        inv_K_all = torch.linalg.inv(K_all)
        K_1, inv_K_1 = K_all[self._ei], inv_K_all[self._ei]
        K_2, inv_K_2 = K_all[self._ej], inv_K_all[self._ej]
        depth_all = torch.stack(self.get_depthmaps(raw=False)).unsqueeze(1)
        depth1, depth2 = depth_all[self._ei], depth_all[self._ej]
        disp_1, disp_2 = 1 / (depth1 + 1e-6), 1 / (depth2 + 1e-6)
        ego_flow_1_2, _ = self.depth_wrapper(R1, T1, R2, T2, disp_1, K_2, inv_K_1)
        ego_flow_2_1, _ = self.depth_wrapper(R2, T2, R1, T1, disp_2, K_1, inv_K_2)
        
        dynamic_masks_all = torch.stack(self.dynamic_masks).to(self.device).unsqueeze(1).bool()
        dynamic_mask1, dynamic_mask2 = dynamic_masks_all[self._ei], dynamic_masks_all[self._ej]
        flow_loss_i = self.flow_loss_fn(ego_flow_1_2[:, :2, ...], self.flow_ij, ~dynamic_mask1, per_pixel_thre=self.pxl_thre)
        flow_loss_j = self.flow_loss_fn(ego_flow_2_1[:, :2, ...], self.flow_ji, ~dynamic_mask2, per_pixel_thre=self.pxl_thre)
        flow_loss = flow_loss_i + flow_loss_j

        if flow_loss.item() > 25: 
            flow_loss = flow_loss * 0.0
        
        return flow_loss


def _fast_depthmap_to_pts3d(depth, pixel_grid, focal, pp):
    pp = pp.unsqueeze(1)
    focal = focal.unsqueeze(1)
    assert focal.shape == (len(depth), 1, 1)
    assert pp.shape == (len(depth), 1, 2)
    assert pixel_grid.shape == depth.shape + (2,)
    depth = depth.unsqueeze(-1)
    return torch.cat((depth * (pixel_grid - pp) / focal, depth), dim=-1)

def _fast_pts3d_to_2d(pts3d, focal, pp):
    pp = pp.unsqueeze(1)
    focal = focal.unsqueeze(1)

    pts2d = (pts3d[..., :2] * focal) / pts3d[..., 2:3] + pp
    return pts2d 


def ParameterStack(params, keys=None, is_param=None, fill=0):
    if keys is not None:
        params = [params[k] for k in keys]

    if fill > 0:
        params = [_ravel_hw(p, fill) for p in params]

    requires_grad = params[0].requires_grad
    assert all(p.requires_grad == requires_grad for p in params)

    params = torch.stack(list(params)).float().detach()
    if is_param or requires_grad:
        params = nn.Parameter(params)
        params.requires_grad_(requires_grad)
    return params


def _ravel_hw(tensor, fill=0):
    # ravel H,W
    tensor = tensor.view((tensor.shape[0] * tensor.shape[1],) + tensor.shape[2:])

    if len(tensor) < fill:
        tensor = torch.cat((tensor, tensor.new_zeros((fill - len(tensor),)+tensor.shape[1:])))
    return tensor


def acceptable_focal_range(H, W, minf=0.5, maxf=3.5):
    focal_base = max(H, W) / (2 * np.tan(np.deg2rad(60) / 2))  # size / 1.1547005383792515
    return minf*focal_base, maxf*focal_base


def apply_mask(img, msk):
    img = img.copy()
    img[msk] = 0
    return img
