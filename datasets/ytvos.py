# ------------------------------------------------------------------------
# InstanceFormer Data Loader.
# ------------------------------------------------------------------------
# Modified from SeqFormer (https://github.com/wjf5203/SeqFormer)
# Copyright (c) 2021 Junfeng Wu. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable VisTR (https://github.com/Epiphqny/VisTR)
# ------------------------------------------------------------------------


from pathlib import Path
import torch
import torch.utils.data
import torchvision
from pycocotools.ytvos import YTVOS
from pycocotools.ytvoseval import YTVOSeval
import datasets.transforms_clip as T
from pycocotools import mask as coco_mask
import os
from PIL import Image
from random import randint
import cv2
import random
import numpy as np
from util.misc import valid_objs
import torchvision.transforms as TT
from inference import CLASSES,CLASSES_YVIS_22


class YTVOSDataset:
    def __init__(self, img_folder, ann_file, transforms, args = None):
        self.img_folder = img_folder
        self.ann_file = ann_file
        self._transforms = transforms
        self.num_classes = args.num_classes
        self.return_masks = args.masks
        self.num_frames = args.num_frames
        self.hidden_dim = args.hidden_dim
        self.prepare = ConvertCocoPolysToMask(args.masks)
        self.ytvos = YTVOS(ann_file)
        self.cat_ids = self.ytvos.getCatIds()
        self.vid_ids = self.ytvos.getVidIds()
        self.vid_infos = []
        self.mtoken = args.memory_token
        self.mframe = args.memory_support
        for i in self.vid_ids:
            info = self.ytvos.loadVids([i])[0]
            info['filenames'] = info['file_names']
            self.vid_infos.append(info)
        self.img_ids = []
        for idx, vid_info in enumerate(self.vid_infos):
            for frame_id in range(len(vid_info['filenames'])):
                self.img_ids.append((idx, frame_id))

        print('\n video num:', len(self.vid_ids), '  clip num:', len(self.img_ids))
        print('\n')

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        instance_check = False
        while not instance_check:
            vid, frame_id = self.img_ids[idx]
            vid_len = len(self.vid_infos[vid]['file_names'])
            num_frames = self.num_frames
            # random sparse sample
            sample_indx = [frame_id]
            #local sample
            samp_id_befor = randint(1,3)
            samp_id_after = randint(1,3)
            local_indx = [max(0, frame_id - samp_id_befor), min(vid_len - 1, frame_id + samp_id_after)]
            sample_indx.extend(local_indx)

            # global sampling
            if num_frames > 3:
                all_inds = list(range(vid_len))
                global_inds = all_inds[:min(sample_indx)]+all_inds[max(sample_indx):]
                global_n = num_frames - len(sample_indx)
                if len(global_inds) > global_n:
                    select_id = random.sample(range(len(global_inds)),global_n)
                    for s_id in select_id:
                        sample_indx.append(global_inds[s_id])
                elif vid_len >=global_n:  # sample long range global frames
                    select_id = random.sample(range(vid_len),global_n)
                    for s_id in select_id:
                        sample_indx.append(all_inds[s_id])
                else:
                    select_id = random.sample(range(vid_len),global_n - vid_len)+list(range(vid_len))
                    for s_id in select_id:
                        sample_indx.append(all_inds[s_id])
            sample_indx.sort()

            vid_id = self.vid_infos[vid]['id']
            img = []
            inds = list(range(self.num_frames))
            img_paths = []
            for j in range(self.num_frames):
                img_path = os.path.join(str(self.img_folder), self.vid_infos[vid]['file_names'][sample_indx[j]])
                img.append(Image.open(img_path).convert('RGB'))
                img_paths.append(img_path)

            ann_ids = self.ytvos.getAnnIds(vidIds=[vid_id])
            target = self.ytvos.loadAnns(ann_ids)

            target = {'video_id': vid, 'annotations': target}
            target_inds = inds
            target = self.prepare(img[0], target, target_inds, sample_inds = sample_indx)

            if self._transforms is not None:
                img, target = self._transforms(img, target, num_frames)

            if len(target['labels']) == 0: # None instance 
                idx = random.randint(0,self.__len__()-1)
            else:
                instance_check=True

        target['boxes'] = target['boxes'].view(len(target['labels']),self.num_frames,4).permute(1,0,2) # for (#frames,#num_of_instances,4)
        target['masks'] = target['masks'].view(len(target['labels']),self.num_frames,target['size'][0], target['size'][1]).permute(1,0,2,3) # (#frames,#num_of_instances,,h,w)

        #objs are not valid before they enter or after it leaves the frame, while in occlusion are valid
        target = valid_objs(self,target)

        target['boxes']=target['boxes'].clamp(1e-6)
        return torch.cat(img, dim=0), target, img_paths


def convert_coco_poly_to_mask(segmentations, height, width, is_crowd):
    masks = []
    for i, seg in enumerate(segmentations):
        if not seg:
            mask = torch.zeros((height,width), dtype=torch.uint8)
        else:
            if not is_crowd[i]:
                seg = coco_mask.frPyObjects(seg, height, width)
            mask = coco_mask.decode(seg)
            if len(mask.shape) < 3:
                mask = mask[..., None]
            mask = torch.as_tensor(mask, dtype=torch.uint8)
            mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target, target_inds, sample_inds):
        w, h = image.size
        video_id = target['video_id']

        anno = target["annotations"]
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]
        video_len = len(anno[0]['bboxes'])
        boxes = []
        classes = []
        segmentations = []
        area = []
        iscrowd = []
        valid = []
        # add valid flag for bboxes
        for i, ann in enumerate(anno):
            classes.append(ann["category_id"])
            for id in target_inds:
                bbox = ann['bboxes'][sample_inds[id]]
                areas = ann['areas'][sample_inds[id]]
                segm = ann['segmentations'][sample_inds[id]]
                # for empty boxes
                if bbox is None: # todo replace this with hungariean matching
                    bbox = [0,0,0,0]
                    areas = 0
                    valid.append(0)
  
                else:
                    valid.append(1)
                crowd = ann["iscrowd"] if "iscrowd" in ann else 0
                boxes.append(bbox)
                area.append(areas)
                segmentations.append(segm)
                # classes.append(clas)
                iscrowd.append(crowd)
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        classes = torch.tensor(classes, dtype=torch.int64)
        if self.return_masks:
            masks = convert_coco_poly_to_mask(segmentations, h, w, iscrowd)
        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks

        image_id = [sample_inds[id] + video_id * 1000 for id in target_inds]
        # for id in image_id:
        #     if id in [1992004, 1992005, 1992006, 1992018]:
        #         print()
        image_id = torch.tensor(image_id)
        target["image_id"] = image_id

        # for conversion to coco api
        area = torch.tensor(area) 
        iscrowd = torch.tensor(iscrowd)
        target["valid"] = torch.tensor(valid)
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        return  target


def make_coco_transforms(image_set, debug=False, memory=False):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    scales = [288, 320, 352, 392, 416, 448, 480, 512]
    if image_set == 'train' and not memory:
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.PhotometricDistort(),
            T.RandomSelect(
                T.Compose([
                    T.RandomResize(scales, max_size= 400 if debug else 768),
                    T.Check(),
                ]),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=400 if debug else 768),
                    T.Check(),
                ])
            ),
            normalize,
        ])

    if image_set == 'val' or memory:
        return T.Compose([
            T.RandomResize([360], max_size=640 if image_set == 'val' else None),
            normalize,
        ])
        
    raise ValueError(f'unknown {image_set}')




def build(image_set, args):
    root = Path(args.ytvis_path)
    assert root.exists(), f'provided YTVOS path {root} does not exist'

    if args.dataset_file == 'YoutubeVIS' or args.dataset_file == 'jointcoco':
        mode = 'instances'
        PATHS = {
            "train": (root / "train/JPEGImages", root / "annotations" / f'{mode}_train_sub.json'),
            "val": (root / "valid/JPEGImages", root / "annotations" / f'{mode}_val_sub.json'),
        }
        img_folder, ann_file = PATHS[image_set]
        print('use Youtube-VIS dataset')
        dataset = YTVOSDataset(img_folder, ann_file, transforms=make_coco_transforms(image_set,debug=args.debug), args=args)

    return dataset
