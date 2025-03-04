# This part is the implementation of part A.6 APPLICATION: Dynamic Video Pose Estimation from our [paper](https://arxiv.org/abs/2410.24211)


1. Run 3D dense track at multiple keyframes (we sample keyframes with a stride of 2 to reduce computational cost):

```bash
  python3 sfm/run_densetrack_multiframes.py --ckpt checkpoints/densetrack3d.pth --video_path demo_data/yellow-duck --output_path results/demo # run with Unidepth
```

2. Use GroundedSAM2 to obtain foreground/dynamic masks from input video

3. Run global alignment (similar to Dust3r/Monst3r) to obtain global depthmaps + camera poses:

```bash
  python3 sfm/run_global_alignment.py --video_path demo_data/yellow-duck --output_path results/demo # run with Unidepth
```