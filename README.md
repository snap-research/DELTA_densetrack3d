<h1 align="center">
  DELTA: Dense Efficient Long-range 3D Tracking for Any video
</h1>

This is the official GitHub repository of the paper:

**[DELTA: Dense Efficient Long-range 3D Tracking for Any video](https://snap-research.github.io/DELTA/)**
</br>
[Tuan Duc Ngo](https://ngoductuanlhp.github.io/),
[Peiye Zhuang](https://payeah.net/),
[Chuang Gan](https://people.csail.mit.edu/ganchuang/),
[Evangelos Kalogerakis](https://kalo-ai.github.io/),
[Sergey Tulyakov](http://www.stulyakov.com/),
[Hsin-Ying Lee](http://hsinyinglee.com/),
[Chaoyang Wang](https://mightychaos.github.io/),
</br>
*ICLR 2025*

### [Project Page](https://snap-research.github.io/DELTA/) | [Paper](https://arxiv.org/abs/2410.24211) | [BibTeX](#citing-delta)


<img width="1100" src="./assets/teaser.png" />

**DELTA** captures **dense, long-range, 3D trajectories** from casual videos in a **feed-forward** manner.


## TODO
- [x] Release model weights on [Google Drive](https://drive.google.com/file/d/18d5M3nl3AxbG4ZkT7wssvMXZXbmXrnjz/view?usp=sharing) and [demo script](demo.py)
- [x] Release training code & dataset preparation
- [x] Release evaluation code
- [ ] Gradio Demo

## Getting Started

### Installation

1. Clone DELTA.
```bash
git clone --recursive https://github.com/snap-research/DenseTrack3D
cd DenseTrack3D
## if you have already cloned DenseTrack3D:
# git submodule update --init --recursive
```

2. Create the environment.
```bash
conda create -n densetrack3d python=3.10 cmake=3.14.0 -y # we recommend using python<=3.10
conda activate densetrack3d 
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y  # use the correct version of cuda for your system

pip install pip==24.0 # downgrade pip to install pytorch_lightning==1.6.0
pip3 install -r requirements.txt
conda install ffmpeg -c conda-forge # to write .mp4 video

pip3 install -U "ray[default]" # for parallel processing
pip3 install viser # for visualize 3D trajectories
```

3. Install `Unidepth`.
```bash
pip3 install ninja
pip3 install -v -U git+https://github.com/facebookresearch/xformers.git@v0.0.24 # Unidepth requires xformers==0.0.2
```

4. [Optional] Install `viser` and `open3d` for 3D visualization.

```bash
pip3 install viser
pip3 install open3d
```

5. [Optional] Install dependencies to generate training data with [Kubric](https://github.com/google-research/kubric).

```bash
pip3 install bpy==3.4.0
pip3 install pybullet
pip3 install OpenEXR
pip3 install tensorflow tensorflow-datasets>=4.1.0 tensorflow-graphics

cd data/kubric/
pip install -e .
cd ../..
```


### Download Checkpoints

The pretrained checkpoints can be downloaded on [Google Drive](https://drive.google.com/file/d/18d5M3nl3AxbG4ZkT7wssvMXZXbmXrnjz/view?usp=sharing).


Run the following commands to download:
```bash
# download the weights
mkdir -p ./checkpoints/
gdown --fuzzy https://drive.google.com/file/d/18d5M3nl3AxbG4ZkT7wssvMXZXbmXrnjz/view?usp=sharing -O ./checkpoints/
```


### Inference

1. We provide 3 sample videos (`car-roundabout`, `rollerblade` from DAVIS, and `yellow-duck` generated by SORA). To run the inference code, you can use the following command:

```bash
python3 demo.py --ckpt checkpoints/densetrack3d.pth --video_path demo_data/yellow-duck --output_path results/demo # run with Unidepth

# or
python3 demo.py --ckpt checkpoints/densetrack3d.pth --video_path demo_data/yellow-duck --output_path results/demo --use_depthcrafter # run with DepthCrafter
```

By default, densely tracking a video of ~100 frames requires ~40GB of GPU memory. To reduce memory consumption, we can use a larger upsample factor (e.g., 8x) and enable fp16 inference, which reduces the requirement to ~20GB of GPU memory:

```bash
python3 demo.py --upsample_factor 8 --use_fp16 --ckpt checkpoints/densetrack3d.pth --video_path demo_data/yellow-duck --output_path results/demo
```

2. Visualize the dense 3D tracks with `viser`:

```bash
python3 visualizer/vis_densetrack3d.py --filepath results/demo/yellow-duck/dense_3d_track.pkl
```

3. [Optional] Visualize the dense 3D tracks with `open3d` (GUI required). To highlight the trajectories of the foreground object, we provide a binary foreground mask for the first frame of the video (the starting frame for dense tracking), which can be obtained with SAM.

```bash
# first run with mode=choose_viewpoint, a 3D GUI will pop-up and you can select the proper viewpoint to capture. Press "S" to save the viewpoint and exit.
python3 visualizer/vis_open3d.py --filepath results/yellow-duck/dense_3d_track.pkl --fg_mask_path demo_data/yellow-duck/yellow-duck_mask.png --video_name yellow-duck --mode choose_viewpoint

# Then run with mode=capture to start rendering 2D video of dense tracking
python3 visualizer/vis_open3d.py --filepath results/yellow-duck/dense_3d_track.pkl --fg_mask_path demo_data/yellow-duck/yellow-duck_mask.png --video_name yellow-duck --mode capture
```

### Prepare training & evaluation data
Please follow the instructions [here](data/DATA_PREPARATION.md) to prepare the training & evaluation data

### Training

1. Pretrain dense 2D tracking model

```bash
bash scripts/train/pretrain_2d.sh
```

2. Train dense 3D tracking model

```bash
bash scripts/train/train.sh
```

### Evaluation 

1. Evaluate sparse 3D tracking on the [TAPVid3D Benchmark](https://tapvid3d.github.io/)

```bash
# Note: replace TAPVID3D_DIR with the real path to tapvid3d dataset
python3 scripts/eval/eval_3d.py
```

2. Evaluate dense 2D tracking on the [CVO Benchmark](https://16lemoing.github.io/dot/)

```bash
# Note: replace CVO_DIR with the real path to CVO dataset
python3 scripts/eval/eval_flow2d.py
```

3. Evaluate sparse 2D tracking on the [TAPVid2D Benchmark](https://tapvid.github.io/)

```bash
# Note: replace TAPVID2D_DIR with the real path to tapvid2d dataset
python3 scripts/eval/eval_2d.py
```



## Citing DELTA

If you find our repository useful, please consider giving it a star ⭐ and citing our paper in your work:

```bibtex
@article{ngo2024delta,
  author    = {Ngo, Tuan Duc and Zhuang, Peiye and Gan, Chuang and Kalogerakis, Evangelos and Tulyakov, Sergey and Lee, Hsin-Ying and Wang, Chaoyang},
  title     = {DELTA: Dense Efficient Long-range 3D Tracking for Any video},
  journal   = {arXiv preprint arXiv:2410.24211},
  year      = {2024}
}
```
