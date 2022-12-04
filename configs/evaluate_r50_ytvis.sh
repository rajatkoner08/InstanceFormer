#!/usr/bin/env bash
# To pretrain, train and evaluate on Swin backbone use --backbone swin_l_p4w12

set -x

python -u inference.py \
    --img_path path_to_ytvis_[19/21/22]/valid/JPEGImages \
    --ann_path path_to_ytvis_[19/21/22]/annotations/annotations_valid.json \
    --initial_output_dir your_output_directory \
    --dataset_file jointovis \
    --backbone resnet50 \
    --masks \
    --with_box_refine \
    --rel_coord \
    --weighted_category \
    --exp_name r50_ytvis_[19/21/22] \
    --propagate_ref_points \
    --propagate_class \
    --increase_proposals 5 \
    --cuda_visible_devices 0 \
