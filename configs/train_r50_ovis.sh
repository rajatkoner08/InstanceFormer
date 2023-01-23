#!/usr/bin/env bash
# To pretrain, train and evaluate on Swin backbone use --backbone swin_l_p4w12

set -x

python -u main.py \
    --ytvis_path path_to_ovis \
    --coco_path path_to_coco \
    --num_frames 3 \
    --memory_support 3 \
    --dataset_file jointovis \
    --backbone resnet50 \
    --masks \
    --with_box_refine \
    --rel_coord \
    --epochs 16 \
    --lr 2e-4 \
    --lr_drop 4 10 \
    --exp_name r50_ovis \
    --batch_size 4 \
    --pretrain_weights path_to_r50_ytvis19_weight \
    --propagate_ref_points \
    --propagate_class \
    --skip_pretrain_detr_temporal_class \
    --memory_token 10 \
    --supcon \
    --cuda_visible_device 0 1 2 3 \

