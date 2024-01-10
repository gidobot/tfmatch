export CUDA_VISIBLE_DEVICES=0 && python train_keras.py \
    --train_config configs/train_keras_config.yaml \
    --gl3d ../GL3D \
    --save_dir ckpt-contextdesc \
    --is_training=True --device_idx 0 --data_split gl3d --dry_run=False