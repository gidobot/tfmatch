export CUDA_VISIBLE_DEVICES='' && python train.py \
    --train_config configs/train_contextdesc_config.yaml \
    --gl3d ../GL3D \
    --save_dir ckpt-contextdesc \
    --is_training=True --device_idx 0 --data_split gl3d