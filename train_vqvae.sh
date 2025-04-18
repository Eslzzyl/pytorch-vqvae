#! /bin/bash

export HF_ENDPOINT=https://hf-mirror.com
uv run vqvae.py \
    --data-folder ./dataset \
    --dataset tinyimagenet \
    --logs-folder /root/tf-logs \
    --hidden-size 128 \
    --k 256 \
    --batch-size 256 \
    --num-epochs 100 \
    --lr 2e-4 \
    --beta 1.0 \
    --num-workers 8