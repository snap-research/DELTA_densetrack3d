# TRAINING DATA PREPARATION

We use [Kubric](https://github.com/google-research/kubric) as our training data

1. Setup `gsutil` using this [instruction](https://cloud.google.com/storage/docs/gsutil_install#linux)

2. Download raw data from [GCD](https://console.cloud.google.com/storage/browser/kubric-public/tfds/movi_f/512x512?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22)))

```bash
mkdir -p datasets/movi_f/
gsutil -m cp -r gs://kubric-public/tfds/movi_f/512x512 ./datasets/movi_f/
```

3. Generate dense 3D annotations

```bash
cd data/kubric/challenges/point_tracking

python3 dataset_mix.py --raw_dir ./datasets/movi_f/ --processed_dir ./datasets/kubric_processed_mix_3d --split train

python3 dataset_mix.py --raw_dir ./datasets/movi_f/ --processed_dir ./datasets/kubric_processed_mix_3d --split validation

```


# EVALUATION DATA PREPARATION

## TAPVid3D
TAPVid3D contains more than 4,000 real-world videos from Aria Digital Twin, Waymo, and Panoptic Studio datasets. Please follow the instruction [here](https://github.com/google-deepmind/tapnet/tree/main/tapnet/tapvid3d) to download and preprocess the data.

Then run the following command to obtain the videodepths from Unidepth/Zoedepth:

```bash
python3 preprocess/preprocess_tapvid3d.py --dataset_root <PATH_TO_TAPVID3D> --split <adt/drivetrack/pstudio>
```

## TAPVid2D
We utilize 3 subsets from the TAPVid2D benchmark, including 30 videos from the DAVIS val set, 1000 videos from the Kinetics val set, 50 synthetic Deepmind Robotics videos. Please follow the instruction [here](https://github.com/google-deepmind/tapnet/tree/main/tapnet/tapvid) to download and preprocess the data.

Then run the following command to obtain the videodepths from Unidepth/Zoedepth:

```bash
python3 preprocess/preprocess_tapvid2d.py --dataset_root <PATH_TO_TAPVID2D> --split <davis/rgb_stacking/kinetics> --use_zoedepth True 
```

## CVO
Download data from [here](https://github.com/16lemoing/dot), including the `clean`, `final` and `extended` subsets.

Then run the following command to obtain the videodepths from Unidepth/Zoedepth:

```bash
python3 preprocess/preprocess_cvo.py --dataset_root <PATH_TO_TAPVID2D> --split <clean/final/extended>
```

## LSFOdyssey
Download data from [here](https://github.com/wwsource/SceneTracker)


# DEMO DATA
Most of our demo videos are taken from the [DAVIS dataset](https://davischallenge.org/index.html) and [SORA](https://openai.com/sora/).