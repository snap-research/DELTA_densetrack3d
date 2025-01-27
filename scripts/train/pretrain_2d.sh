python scripts/train/train_2d.py \
    --batch_size 1 \
    --num_steps 100000 \
    --ckpt_path logdirs/pretrain_densetrack2d \
    --model_name densetrack3d \
    --save_freq 500 \
    --eval_datasets tapvid_davis_first \
    --traj_per_sample 256 \
    --sliding_window_len 16 \
    --num_virtual_tracks 64 \
    --model_stride 4 \
    --lr 0.0001 \
    --evaluate_every_n_epoch 1 \
    --sequence_len 24 \
    --num_nodes 1 \
    --lambda_2d 100.0 \
    --lambda_vis 1.0 \
    --lambda_conf 1.0 
