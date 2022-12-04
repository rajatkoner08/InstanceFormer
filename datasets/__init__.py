# ------------------------------------------------------------------------
# InstanceFormer Transformer classes.
# ------------------------------------------------------------------------
# Modified from SeqFormer (https://github.com/wjf5203/SeqFormer)
# Copyright (c) 2021 Junfeng Wu. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

import torch.utils.data
from .torchvision_datasets import CocoDetection
from datasets.ytvos import YTVOSDataset as YTVOSDataset

from .coco import build as build_coco
from .coco2seq import build as build_seq_coco
from .concat_dataset import build as build_joint
from .ytvos import build as build_ytvs
from .ovis import build as build_ovis


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, CocoDetection):
        return dataset.coco
    if isinstance(dataset, YTVOSDataset):
        return dataset.ytvos


def build_dataset(image_set, args):
    if args.dataset_file == 'YoutubeVIS':
        return build_ytvs(image_set, args)
    if args.dataset_file == 'ovis':
        return build_ovis(image_set, args)
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    if args.dataset_file == 'Seq_coco':
        return build_seq_coco(image_set, args)
    if 'joint' in args.dataset_file:
        return build_joint(image_set, args)

    raise ValueError(f'dataset {args.dataset_file} not supported')

