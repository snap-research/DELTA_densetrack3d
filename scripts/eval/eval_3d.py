import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


import json
import time
from dataclasses import dataclass, field

import hydra
import numpy as np
import ray
import torch
from densetrack3d.datasets.dr_dataset import DynamicReplicaDataset
from densetrack3d.datasets.lsfodyssey_dataset import LSFOdyssey
from densetrack3d.datasets.tapvid3d_dataset import TapVid3DDataset
from densetrack3d.datasets.utils import collate_fn
from densetrack3d.evaluation.core.evaluator import Evaluator

# from densetrack3d.models.densetrack3d.densetrack3d import DenseTrack3D
from densetrack3d.models.densetrack3d.densetrack3d import DenseTrack3D
from densetrack3d.models.evaluation_predictor.evaluation_predictor import EvaluationPredictor
from densetrack3d.utils.io import load_checkpoint
from omegaconf import OmegaConf


TAPVID3D_DIR = None # FIXME: replace the path to the tapvid3d dataset here

@dataclass(eq=False)
class DefaultConfig:
    # Directory where all outputs of the experiment will be saved.
    exp_dir: str = "./results/densetrack3d_tapvid3d"
    # exp_dir: str = "./results/tapvid3d_logratio_origdepth_all"

    # Name of the dataset to be used for the evaluation.
    dataset_name: str = "tapvid3d"
    # dataset_name: str = "lsfodyssey"
    # The root directory of the dataset.
    dataset_root: str = TAPVID3D_DIR

    checkpoint: str = "checkpoints/densetrack3d.pth"
    # EvaluationPredictor parameters
    # The size (N) of the support grid used in the predictor.
    # The total number of points is (N*N).
    grid_size: tuple = (12, 16)
    # The size (N) of the local support grid.
    local_grid_size: int = 8
    # A flag indicating whether to evaluate one ground truth point at a time.
    single_point: bool = False
    # The number of iterative updates for each sliding window.
    n_iters: int = 6

    seed: int = 0
    gpu_idx: int = 0

    # Override hydra's working directory to current working dir,
    # also disable storing the .hydra logs:
    hydra: dict = field(
        default_factory=lambda: {
            "run": {"dir": "."},
            "output_subdir": None,
        }
    )


@ray.remote(num_gpus=1)
class RayPredictor:
    def __init__(
        self,
        worker_id: int = 0,
        cfg: DefaultConfig = None,
    ):
        self.worker_id = worker_id
        self.cfg = cfg

    def run_eval(self, split):
        cfg = self.cfg
        os.makedirs(cfg.exp_dir, exist_ok=True)

        evaluator = Evaluator(cfg.exp_dir)

        model = DenseTrack3D(
            stride=4,
            window_len=16,
            add_space_attn=True,
            num_virtual_tracks=64,
            model_resolution=(384, 512),
        )
        state_dict = load_checkpoint(cfg.checkpoint)
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        # Creating the EvaluationPredictor object
        predictor = EvaluationPredictor(
            model,
            # grid_size=16,
            grid_size=16,
            local_grid_size=cfg.local_grid_size,
            single_point=cfg.single_point,
            n_iters=cfg.n_iters,
        )
        predictor = predictor.eval().cuda()

        test_dataset = TapVid3DDataset(
            data_root=cfg.dataset_root,
            datatype=split,
            use_metric_depth=True,  # FIXME check here
            split="all",
            read_from_s3=False,
        )

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )

        start = time.time()
        evaluate_result = evaluator.evaluate_sequence_3d(
            predictor,
            test_dataloader,
            dataset_name=cfg.dataset_name,
            lift_3d=False,  # FIXME only for ablation
            verbose=True,
        )
        end = time.time()
        print(f"Time taken for evaluation on {split} split: {end - start:.3f} seconds")

        # Saving the evaluation results to a .json file
        # evaluate_result = evaluate_result["avg"]

        saved_evaluate_result = {}
        saved_evaluate_result["time"] = end - start
        saved_evaluate_result["avg"] = evaluate_result["avg"]

        for k, v in evaluate_result.items():
            if "avg" not in k:
                saved_evaluate_result[k] = float(v["average_pts_within_thresh"])
        print("evaluate_result", evaluate_result["avg"])

        result_file = os.path.join(cfg.exp_dir, f"result_eval_{split}.json")

        print(f"Dumping eval results to {result_file}.")
        with open(result_file, "w") as f:
            json.dump(saved_evaluate_result, f, indent=4)


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="default_config_eval", node=DefaultConfig)


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    cfg = DefaultConfig()

    splits = ["adt", "drivetrack", "pstudio"]

    num_gpus = min(len(splits), torch.cuda.device_count())

    print(f"Using {num_gpus} GPUs")

    ray.init()
    predictors = [RayPredictor.remote(i, cfg) for i in range(num_gpus)]
    tasks = [predictor.run_eval.remote(split) for predictor, split in zip(predictors, splits)]

    ray.get(tasks)
    ray.shutdown()
