import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from densetrack3d.datasets.cvo_dataset import CVO
from densetrack3d.datasets.kubric_dataset import KubricDataset
from densetrack3d.datasets.tapvid2d_dataset import TapVid2DDataset
from densetrack3d.datasets.utils import collate_fn, collate_fn_train, dataclass_to_cuda_
from densetrack3d.evaluation.core.evaluator import Evaluator
from densetrack3d.models.densetrack3d.densetrack2d import DenseTrack2D
from densetrack3d.models.evaluation_predictor.evaluation_predictor import EvaluationPredictor
from densetrack3d.models.loss import balanced_bce_loss, bce_loss, confidence_loss, track_loss
from densetrack3d.models.model_utils import (
    bilinear_sampler,
    dense_to_sparse_tracks_3d_in_3dspace,
    get_grid,
    get_points_on_a_grid,
)
from densetrack3d.models.optimizer import fetch_optimizer
from densetrack3d.utils.logger import Logger
from densetrack3d.utils.visualizer import Visualizer, flow_to_rgb
from einops import rearrange
from pytorch_lightning.lite import LightningLite

# from torch.cuda.amp import GradScaler
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


# from densetrack3d.utils.signal_utils import sig_handler, term_handler

TAPVID2D_DIR, KUBRIC3D_MIX_DIR = None, None


def sample_sparse_queries(trajs_g, vis_g):
    B, T, N, D = trajs_g.shape
    device = trajs_g.device

    # NOTE sample sparse queries
    __, first_positive_inds = torch.max(vis_g, dim=1)
    # We want to make sure that during training the model sees visible points
    # that it does not need to track just yet: they are visible but queried from a later frame
    N_rand = N // 4
    # inds of visible points in the 1st frame
    nonzero_inds = [[torch.nonzero(vis_g[b, :, i]) for i in range(N)] for b in range(B)]

    for b in range(B):
        rand_vis_inds = torch.cat(
            [nonzero_row[torch.randint(len(nonzero_row), size=(1,))] for nonzero_row in nonzero_inds[b]],
            dim=1,
        )
        first_positive_inds[b] = torch.cat([rand_vis_inds[:, :N_rand], first_positive_inds[b : b + 1, N_rand:]], dim=1)

    ind_array_ = torch.arange(T, device=device)
    ind_array_ = ind_array_[None, :, None].repeat(B, 1, N)
    assert torch.allclose(
        vis_g[ind_array_ == first_positive_inds[:, None, :]],
        torch.ones(1, device=device),
    )
    gather = torch.gather(trajs_g, 1, first_positive_inds[:, :, None, None].repeat(1, 1, N, D))
    xys = torch.diagonal(gather, dim1=1, dim2=2).permute(0, 2, 1)

    sparse_queries = torch.cat([first_positive_inds[:, :, None], xys[:, :, :2]], dim=2)

    return sparse_queries

def forward_batch(batch, model, args):
    video = batch.video
    # videodepth = batch.videodepth
    # depth_init = batch.depth_init

    # max_depth = videodepth[videodepth>0.01].max()

    trajs_g = batch.trajectory
    # trajs_d = batch.trajectory_d
    vis_g = batch.visibility
    valids = batch.valid

    
    # dense_trajectory = batch.dense_trajectory
    # dense_trajectory_d = batch.dense_trajectory_d


    flow = batch.flow
    flow_alpha = batch.flow_alpha


    # breakpoint()
    B, T, C, H, W = video.shape
    assert C == 3
    B, T, N, D = trajs_g.shape
    device = video.device

    sparse_queries = sample_sparse_queries(trajs_g, vis_g)
    n_sparse_queries = sparse_queries.shape[1]

    #############################
    # n_input_queries = 256
    # NOTE add regular grid queries:
    grid_xy = get_points_on_a_grid((12,16), video.shape[3:]).long().float()
    grid_xy = torch.cat([torch.zeros_like(grid_xy[:, :, :1]), grid_xy], dim=2).to(device)  # B, N, C
    # grid_xy_d = bilinear_sampler(depth_init, rearrange(grid_xy[..., 1:3], 'b n c -> b () n c'), mode="nearest")
    # grid_xy_d = rearrange(grid_xy_d, 'b c m n -> b (m n) c')

    grid_queries = grid_xy
    input_queries = torch.cat([sparse_queries, grid_queries], dim=1)

    sparse_predictions, dense_predictions, (sparse_train_data_dict, dense_train_data_dict) = model(
        video=video, 
        # videodepth=videodepth, 
        sparse_queries=input_queries, 
        # depth_init=depth_init,
        iters=args.train_iters,
        is_train=True,
        use_dense=True,
    )

    coord_predictions = sparse_train_data_dict["coords"]
    vis_predictions = sparse_train_data_dict["vis"]
    conf_predictions = sparse_train_data_dict["conf"]
    valid_mask = sparse_train_data_dict["mask"]


    dense_coord_predictions = dense_train_data_dict["coords"] # list of list of (B, T, 2, H, W)
    dense_vis_predictions = dense_train_data_dict["vis"]
    dense_conf_predictions = dense_train_data_dict["conf"]
    (x0, y0) = dense_train_data_dict["x0y0"]

    S = args.sliding_window_len

    seq_loss = torch.tensor(0., requires_grad=True).cuda()
    vis_loss = torch.tensor(0., requires_grad=True).cuda()
    conf_loss = torch.tensor(0., requires_grad=True).cuda()

    for idx, ind in enumerate(range(0, args.sequence_len - S // 2, S // 2)):
        
        traj_gt_ = trajs_g[:, ind : ind + S].clone()
        vis_gt_ = vis_g[:, ind : ind + S].clone()
        valid_gt_ = valids[:, ind : ind + S].clone() * valid_mask[:, ind : ind + S, :n_sparse_queries].clone()

        coord_predictions_ = coord_predictions[idx][:, :, :, :n_sparse_queries, :]
        vis_predictions_ = vis_predictions[idx][:, :, :n_sparse_queries]
        conf_predictions_ = conf_predictions[idx][:, :, :n_sparse_queries]

        coord_predictions_[..., 0] /= (W-1)
        coord_predictions_[..., 1] /= (H-1)

        traj_gt_[..., 0] /= (W-1)
        traj_gt_[..., 1] /= (H-1)



        
        seq_loss += track_loss(coord_predictions_, traj_gt_, valid_gt_)
        vis_loss += balanced_bce_loss(vis_predictions_, vis_gt_, valid_gt_)
        conf_loss += confidence_loss(coord_predictions_, conf_predictions_, traj_gt_, vis_gt_,valid_gt_, expected_dist_thresh=12.0/(W-1))

    

    seq_loss = seq_loss * args.lambda_2d / len(coord_predictions)
    vis_loss = vis_loss * args.lambda_vis / len(coord_predictions)
    conf_loss = conf_loss * args.lambda_conf / len(coord_predictions)


    dense_seq_loss = torch.tensor(0., requires_grad=True).cuda()
    dense_vis_loss = torch.tensor(0., requires_grad=True).cuda()
    dense_conf_loss = torch.tensor(0., requires_grad=True).cuda()

    for idx, ind in enumerate(range(0, args.sequence_len - S // 2, S // 2)):

        if idx >= len(dense_coord_predictions):
            break
        
        dense_coord_prediction_ = dense_coord_predictions[idx][0]
        dense_vis_prediction_ = dense_vis_predictions[idx][0] # T,  H, W
        dense_conf_prediction_ = dense_conf_predictions[idx][0] # T,  H, W

        pred_H, pred_W = dense_coord_prediction_.shape[-2:]

        gt_alpha = flow_alpha[0, ind : ind + S, y0[0]*4:(y0[0]*4 + pred_H), x0[0]*4:(x0[0]*4 + pred_W)].clone() # T 2 H W
        gt_flow = flow[0, ind : ind + S, :, y0[0]*4:(y0[0]*4 + pred_H), x0[0]*4:(x0[0]*4 + pred_W)].clone() # T 2 H_crop W_crop

        I, S = dense_coord_prediction_.shape[:2]

        
        dense_grid_2d = get_grid(pred_H, pred_W, normalize=False, device=dense_coord_prediction_.device) # H W 2
        dense_grid_2d[..., 0] += x0[0]*4
        dense_grid_2d[..., 1] += y0[0]*4
        dense_grid_2d = rearrange(dense_grid_2d, 'h w c -> () c h w')

        gt_coord_ = gt_flow + dense_grid_2d
        gt_coord_[:, 0] /= (W-1)
        gt_coord_[:, 1] /= (H-1)

        dense_coord_prediction_[:, :, 0] /= (W-1)
        dense_coord_prediction_[:, :, 1] /= (H-1)

        dense_seq_loss += track_loss(dense_coord_prediction_, gt_coord_)
        dense_vis_loss += bce_loss(dense_vis_prediction_, gt_alpha)
        dense_conf_loss += confidence_loss(dense_coord_prediction_, dense_conf_prediction_, gt_coord_, gt_alpha, expected_dist_thresh=12.0/(W-1), has_batch_dim=False)

    dense_seq_loss = dense_seq_loss * args.lambda_2d / len(dense_coord_predictions) # FIXME
    dense_vis_loss = dense_vis_loss * args.lambda_vis / len(dense_coord_predictions)
    dense_conf_loss = dense_conf_loss * args.lambda_conf / len(dense_coord_predictions)

    ###############################!SECTION########!SECTION


    output = {
        "flow": {
            "loss": seq_loss.mean(),
            "predictions": sparse_predictions["coords"][0].detach()
            },
        "visibility": {
            "loss": vis_loss.mean(), 
            "predictions": sparse_predictions["vis"][0].detach(),
        },
        "conf": {
            "loss": conf_loss.mean(), 
            "predictions": sparse_predictions["conf"][0].detach(),
        },

        "dense_flow": {
            "loss": dense_seq_loss.mean(),
        },
        
        "dense_visibility": {
            "loss": dense_vis_loss.mean(),
        },
        "dense_conf": {
            "loss": dense_conf_loss.mean(),
        },
    }

    return output



def run_test_eval(evaluator, model, dataloaders, writer, step):
    model.eval()
    predictor = EvaluationPredictor(
        model.module.module,
        grid_size=5,
        local_grid_size=0,
        single_point=False,
        n_iters=6,
    )
    predictor = predictor.eval().cuda()

    for ds_name, dataloader in dataloaders:
        visualize_every = 1
        grid_size = 5
        if ds_name == "dynamic_replica":
            visualize_every = 8
            grid_size = 0
        elif "tapvid" in ds_name:
            visualize_every = 5

        if "tapvid" in ds_name:

            metrics = evaluator.evaluate_sequence(
                predictor,
                dataloader,
                dataset_name="tapvid_davis_first",
                is_sparse=True,
                use_2d_only=True
            )
            
            metrics = {
                f"{ds_name}_avg_OA": metrics["avg"]["occlusion_accuracy"],
                f"{ds_name}_avg_delta": metrics["avg"]["average_pts_within_thresh"],
                f"{ds_name}_avg_Jaccard": metrics["avg"]["average_jaccard"],
            }
        elif "CVO" in ds_name:
            metrics = evaluator.evaluate_flow(
                model=predictor,
                test_dataloader=dataloader,
                split="clean"
            )
            
            metrics = {
                f"{ds_name}_avg_epe_all": metrics["avg"]["epe_all"],
                f"{ds_name}_avg_epe_occ": metrics["avg"]["epe_occ"],
                f"{ds_name}_avg_epe_vis": metrics["avg"]["epe_vis"],
                f"{ds_name}_avg_epe_iou": metrics["avg"]["iou"],
            }

        writer.add_scalars(f"Eval_{ds_name}", metrics, step)


class Logger:
    SUM_FREQ = 100

    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = SummaryWriter(log_dir=os.path.join(args.ckpt_path, "runs"))

    def _print_training_status(self):
        metrics_data = [
            self.running_loss[k] / Logger.SUM_FREQ for k in sorted(self.running_loss.keys())
        ]
        training_str = "[{:6d}] ".format(self.total_steps + 1)
        metrics_str = ("{:10.4f}, " * len(metrics_data)).format(*metrics_data)

        # print the training status
        logging.info(f"Training Metrics ({self.total_steps}): {training_str + metrics_str}")

        if self.writer is None:
            self.writer = SummaryWriter(log_dir=os.path.join(args.ckpt_path, "runs"))

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k] / Logger.SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics, task):
        self.total_steps += 1

        for key in metrics:
            task_key = str(key) + "_" + task
            if task_key not in self.running_loss:
                self.running_loss[task_key] = 0.0

            self.running_loss[task_key] += metrics[key]

        if self.total_steps % Logger.SUM_FREQ == Logger.SUM_FREQ - 1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=os.path.join(args.ckpt_path, "runs"))

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


class Lite(LightningLite):
    def run(self, args):
        def seed_everything(seed: int):
            random.seed(seed)
            os.environ["PYTHONHASHSEED"] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        seed_everything(0)

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(0)
        if self.global_rank == 0:
            eval_dataloaders = []
            
            eval_davis_dataset = TapVid2DDataset(
                data_root=TAPVID2D_DIR,
                dataset_type="davis",
                resize_to_256=True,
                queried_first=True,
                read_from_s3=False,
            )

            eval_davis_dataloader = torch.utils.data.DataLoader(
                eval_davis_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=1,
                collate_fn=collate_fn,
            )

            eval_dataloaders.append(("tapvid_davis_first", eval_davis_dataloader))

            save_video_dir = os.path.join(args.ckpt_path, "videos")
            os.makedirs(save_video_dir, exist_ok=True)
            evaluator = Evaluator(save_video_dir)

            visualizer = Visualizer(
                save_dir=args.ckpt_path,
                pad_value=80,
                fps=1,
                show_first_frame=0,
                tracks_leave_trace=0,
            )

        model = DenseTrack2D(
            stride=args.model_stride,
            window_len=args.sliding_window_len,
            add_space_attn=not args.remove_space_attn,
            num_virtual_tracks=args.num_virtual_tracks,
            model_resolution=args.crop_size,
        )

        with open(args.ckpt_path + "/meta.json", "w") as file:
            json.dump(vars(args), file, sort_keys=True, indent=4)


        model.cuda()

        train_dataset = KubricDataset(
            data_root=KUBRIC3D_MIX_DIR,
            crop_size=(384,512),
            seq_len=args.sequence_len,
            traj_per_sample=256,
            sample_vis_1st_frame=True,
            use_augs=not args.dont_use_augs,
            use_gt_depth=True,
        )


        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=True,
            collate_fn=collate_fn_train,
            drop_last=True,
        )

        train_loader = self.setup_dataloaders(train_loader, move_to_device=False)
        print("LEN TRAIN LOADER", len(train_loader))
        optimizer, scheduler = fetch_optimizer(args, model)

        total_steps = 0
        if self.global_rank == 0:
            logger = Logger(model, scheduler)

        folder_ckpts = [
            f
            for f in os.listdir(args.ckpt_path)
            if not os.path.isdir(f) and f.endswith(".pth") and not "final" in f
        ]
        if len(folder_ckpts) > 0:
            ckpt_path = sorted(folder_ckpts)[-1]
            ckpt = self.load(os.path.join(args.ckpt_path, ckpt_path))
            logging.info(f"Loading checkpoint {ckpt_path}")
            if "model" in ckpt:
                model.load_state_dict(ckpt["model"])
            else:
                model.load_state_dict(ckpt)
            if "optimizer" in ckpt:
                logging.info("Load optimizer")
                optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                logging.info("Load scheduler")
                scheduler.load_state_dict(ckpt["scheduler"])
            if "total_steps" in ckpt:
                total_steps = ckpt["total_steps"]
                logging.info(f"Load total_steps {total_steps}")
                # scheduler.total_steps = args.num_steps // args.accum
                # scheduler.step = total_steps // args.accum

                # print('debug', scheduler.step)

        elif args.restore_ckpt is not None:
            assert args.restore_ckpt.endswith(".pth") or args.restore_ckpt.endswith(".pt")
            logging.info(f"Loading checkpoint {args.restore_ckpt}")

            strict = False
            state_dict = self.load(args.restore_ckpt)
            if "model" in state_dict:
                state_dict = state_dict["model"]

            if list(state_dict.keys())[0].startswith("module."):
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

            model_state_dict = model.state_dict()
            new_state_dict = {}
            for k, v in state_dict.items():
                if k in model_state_dict and v.shape == model_state_dict[k].shape:
                    new_state_dict[k] = v

            model.load_state_dict(new_state_dict, strict=strict)
            logging.info(f"Done loading checkpoint")

        elif args.resume_from is not None:
            ckpt = self.load(args.resume_from)
            logging.info(f"Resume checkpoint {args.resume_from}")
            if "model" in ckpt:
                model.load_state_dict(ckpt["model"])
            else:
                model.load_state_dict(ckpt)
            if "optimizer" in ckpt:
                logging.info("Load optimizer")
                optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                logging.info("Load scheduler")
                scheduler.load_state_dict(ckpt["scheduler"])
            if "total_steps" in ckpt:
                total_steps = ckpt["total_steps"]
                logging.info(f"Load total_steps {total_steps}")

        model, optimizer = self.setup(model, optimizer, move_to_device=False)
        # model.cuda()
        model.train()

        save_freq = args.save_freq
        scaler = GradScaler(device="cuda", enabled=False)

        should_keep_training = True
        global_batch_num = 0
        epoch = -1

        # run_test_eval(
        #     evaluator,
        #     model,
        #     eval_dataloaders,
        #     logger.writer,
        #     total_steps,
        # )
        # model.train()
        # torch.cuda.empty_cache()

        while should_keep_training:
            epoch += 1
            for i_batch, batch in enumerate(tqdm(train_loader)):
                batch, gotit = batch
                if not all(gotit):
                    print("batch is None")
                    continue
                dataclass_to_cuda_(batch)

                optimizer.zero_grad()

                assert model.training

                output = forward_batch(batch, model, args)

                loss = 0
                for k, v in output.items():
                    if "loss" in v:
                        loss += v["loss"]


                if self.global_rank == 0:
                    for k, v in output.items():
                        if "loss" in v:
                            logger.writer.add_scalar(
                                f"live_{k}_loss", v["loss"].item(), total_steps
                            )
                        if "metrics" in v:
                            logger.push(v["metrics"], k)
                    if total_steps % save_freq == save_freq - 1:
                        visualizer.visualize(
                            video=batch.video.clone(),
                            tracks=batch.trajectory.clone(),
                            filename="train_gt_traj",
                            writer=logger.writer,
                            step=total_steps,
                        )

                        visualizer.visualize(
                            video=batch.video.clone(),
                            tracks=output["flow"]["predictions"][None],
                            filename="train_pred_traj",
                            writer=logger.writer,
                            step=total_steps,
                        )

                    if len(output) > 1:
                        logger.writer.add_scalar(f"live_total_loss", loss.item(), total_steps)
                    logger.writer.add_scalar(
                        f"learning_rate", optimizer.param_groups[0]["lr"], total_steps
                    )
                    global_batch_num += 1
                self.barrier()

                # breakpoint()
                self.backward(scaler.scale(loss))

                # breakpoint()

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)

                scaler.step(optimizer)
                scheduler.step()
                scaler.update()

                
                total_steps += 1

                if self.global_rank == 0:
                    if (i_batch >= len(train_loader) - 1) or (
                        total_steps == 1 and args.validate_at_start
                    ):
                        if (epoch + 1) % args.save_every_n_epoch == 0:
                            ckpt_iter = "0" * (6 - len(str(total_steps))) + str(total_steps)
                            save_path = Path(
                                f"{args.ckpt_path}/model_{args.model_name}_{ckpt_iter}.pth"
                            )

                            save_dict = {
                                "model": model.module.module.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "scheduler": scheduler.state_dict(),
                                "total_steps": total_steps,
                            }

                            logging.info(f"Saving file {save_path}")
                            self.save(save_dict, save_path)

                        if (epoch + 1) % args.evaluate_every_n_epoch == 0 or (
                            args.validate_at_start and epoch == 0
                        ):
                            run_test_eval(
                                evaluator,
                                model,
                                eval_dataloaders,
                                logger.writer,
                                total_steps,
                            )
                            model.train()
                            torch.cuda.empty_cache()

                self.barrier()
                if total_steps > args.num_steps:
                    should_keep_training = False
                    break
        if self.global_rank == 0:
            print("FINISHED TRAINING")

            PATH = f"{args.ckpt_path}/{args.model_name}_final.pth"
            torch.save(model.module.module.state_dict(), PATH)
            run_test_eval(evaluator, model, eval_dataloaders, logger.writer, total_steps)
            logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="densetrack3d", help="model name")
    parser.add_argument("--restore_ckpt", help="path to restore a checkpoint")
    parser.add_argument("--ckpt_path", help="path to save checkpoints")
    parser.add_argument(
        "--batch_size", type=int, default=4, help="batch size used during training."
    )
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=10, help="number of dataloader workers")

    parser.add_argument("--mixed_precision", action="store_true", help="use mixed precision")
    parser.add_argument("--lr", type=float, default=0.0005, help="max learning rate.")
    parser.add_argument("--wdecay", type=float, default=0.00001, help="Weight decay in optimizer.")
    parser.add_argument(
        "--num_steps", type=int, default=200000, help="length of training schedule."
    )
    parser.add_argument(
        "--evaluate_every_n_epoch",
        type=int,
        default=1,
        help="evaluate during training after every n epochs, after every epoch by default",
    )
    parser.add_argument(
        "--save_every_n_epoch",
        type=int,
        default=1,
        help="save checkpoints during training after every n epochs, after every epoch by default",
    )
    parser.add_argument(
        "--validate_at_start",
        action="store_true",
        help="whether to run evaluation before training starts",
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=100,
        help="frequency of trajectory visualization during training",
    )
    parser.add_argument(
        "--traj_per_sample",
        type=int,
        default=768,
        help="the number of trajectories to sample for training",
    )
    parser.add_argument("--dataset_root", type=str, default="None", help="path lo all the datasets (train and eval)")

    parser.add_argument(
        "--train_iters",
        type=int,
        default=4,
        help="number of updates to the disparity field in each forward pass.",
    )
    parser.add_argument("--sequence_len", type=int, default=24, help="train sequence length")
    parser.add_argument(
        "--eval_datasets",
        nargs="+",
        default=["tapvid_davis_first"],
        help="what datasets to use for evaluation",
    )

    parser.add_argument(
        "--num_virtual_tracks",
        type=int,
        default=64,
        help="Num of virtual tracks",
    )
    parser.add_argument(
        "--dont_use_augs",
        action="store_true",
        help="don't apply augmentations during training",
    )
    parser.add_argument(
        "--sample_vis_1st_frame",
        action="store_true",
        help="only sample trajectories with points visible on the first frame",
    )
    parser.add_argument(
        "--sliding_window_len",
        type=int,
        default=16,
        help="length of the sliding window",
    )
    parser.add_argument(
        "--model_stride",
        type=int,
        default=4,
        help="stride of the feature network",
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        nargs="+",
        default=[384, 512],
        help="crop videos to this resolution during training",
    )

    parser.add_argument(
        "--eval_max_seq_len",
        type=int,
        default=1000,
        help="maximum length of evaluation videos",
    )

    parser.add_argument(
        "--accum",
        type=int,
        default=1,
        help="maximum length of evaluation videos",
    )

    parser.add_argument(
        "--resume_from", 
        default=None, 
        help="path to restore a checkpoint"
    )

    parser.add_argument(
        "--lambda_2d",
        type=float,
        default=10.0,
        help="weight of 2d loss",
    )

    parser.add_argument(
        "--lambda_d",
        type=float,
        default=1.0,
        help="weight of depth loss",
    )

    parser.add_argument(
        "--lambda_vis",
        type=float,
        default=0.1,
        help="weight of vis loss",
    )

    parser.add_argument(
        "--lambda_conf",
        type=float,
        default=0.1,
        help="weight of confidence loss",
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    )

    Path(args.ckpt_path).mkdir(exist_ok=True, parents=True)
    from pytorch_lightning.strategies import DDPStrategy

    Lite(
        strategy=DDPStrategy(find_unused_parameters=False),
        devices="auto",
        accelerator="gpu",
        precision=32,
        num_nodes=args.num_nodes,
    ).run(args)