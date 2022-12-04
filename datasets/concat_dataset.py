# ------------------------------------------------------------------------
# InstanceFormer
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

from pathlib import Path

import torch
import torch.utils.data

from util.misc import get_local_rank, get_local_size
import datasets.transforms_clip as T
from torch.utils.data import Dataset, ConcatDataset
from .coco2seq import build as build_seq_coco
from .ytvos import build as build_ytvs
from .ovis import build as build_ovis


def build(image_set, args):
    print('preparing coco2seq dataset ....')
    coco_seq =  build_seq_coco(image_set, args)

    if 'ovis' in args.dataset_file:
        print('preparing ovis dataset  .... ')
        ovis_dataset = build_ovis(image_set, args)
        concat_data = ConcatDataset([ovis_dataset, coco_seq])
    else:
        print('preparing ytvis dataset  .... ')
        ytvis_dataset = build_ytvs(image_set, args)
        if image_set=='val':
            return ytvis_dataset
        concat_data = ConcatDataset([ coco_seq, ytvis_dataset])

    if args.save2memory: # only for memeory saving, to create large batch processing for coco for faster inference
        return ytvis_dataset, coco_seq
    else:
        return concat_data

