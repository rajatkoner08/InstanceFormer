#!/usr/bin/env bash
# To pretrain, train and evaluate on Swin backbone use --backbone swin_l_p4w12

set -x

python -u main.py \
      --coco_path path_to_coco \
      --num_frames 1 \
      --dataset_file coco \
      --backbone resnet50 \
      --masks \
      --with_box_refine \
      --rel_coord \
      --epochs 12 \
      --lr 1e-4 \
      --lr_drop 4 10 \
      --exp_name r50_pretrain \
      --batch_size 4 \
      --cuda_visible_device 0 1 2 3 \



