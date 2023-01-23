#!/usr/bin/env bash
# To pretrain, train and evaluate on Swin backbone use --backbone swin_l_p4w12

set -x

python -u inference.py \
    --ytvis_path path_to_ovis/valid \
    --ann_path path_to_ovis/annotations_valid.json \
    --initial_output_dir your_output_directory \
    --dataset_file jointovis \
    --backbone resnet50 \
    --masks \
    --with_box_refine \
    --rel_coord \
    --weighted_category \
    --exp_name r50_ovis \
    --propagate_ref_points \
    --propagate_ref_additive \
    --increase_proposals 2 \
    --propagate_class \
    --dynamic_memory \
    --temp_midx_pos \
    --memory_token 10 \
    --memory_support 4 \
    --supcon \
    --cuda_visible_devices 0 \

