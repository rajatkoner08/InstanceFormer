#!/usr/bin/env bash

set -x

python -u main.py \
    --ytvis_path path_to_ytvis_[19/21/22] \
    --coco_path path_to_coco \
    --num_frames 4 \
    --memory_support 4 \
    --dataset_file jointovis \
    --backbone resnet50 \
    --masks \
    --with_box_refine \
    --rel_coord \
    --epochs 16 \
    --lr 1e-4 \
    --lr_drop 4 10 \
    --exp_name r50_ytvis_[19/21/22] \
    --batch_size 4 \
    --pretrain_weights path_to_r50_pretrain_weight \
    --propagate_ref_points \
    --propagate_class \
    --cuda_visible_device 0 1 2 3 \
