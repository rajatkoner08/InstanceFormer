# ------------------------------------------------------------------------
# InstanceFormer Train and Eval functions used in main.py
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

import math
import sys
from typing import Iterable
import cv2
import torch
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.data_prefetcher import data_prefetcher
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

def unwrap(wrapped_module):
    if isinstance(wrapped_module, DistributedDataParallel):
        module = wrapped_module.module
    else:
        module = wrapped_module
    return module

def check_unused_parameters(model, loss_dict, weight_dict):
    print("=== Check unused parameters ===")
    # print unused parameters
    print(f"set(loss_dict) - set(weight_dict) = {set(loss_dict.keys()) - set(weight_dict.keys())}")
    print(f"set(weight_dict) - set(loss_dict) = {set(weight_dict.keys()) - set(loss_dict.keys())}")

    unused_params = [name for name, param in unwrap(model).named_parameters()
                     if param.grad is None and not name.startswith('detr.backbone')]
    if unused_params:
        raise RuntimeError(f"Unused parameters: {unused_params}")
    else:
        print("All the parameters are used.")

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    writer=None, debug=False):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 4000

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    for j in tqdm(metric_logger.log_every(range(len(data_loader)), print_freq, header)):
        outputs, loss_dict = model(samples, targets, criterion)  # freeze_detr=False

        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()

        if j == 0:
            check_unused_parameters(model, loss_dict, weight_dict)

        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)

        optimizer.step()
        if writer != None and j % print_freq == 0:
            itr_count = j+epoch*len(data_loader)
            writer.add_scalar("Loss", loss_value, itr_count)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], itr_count)
            writer.add_scalar("grad_norm", grad_total_norm, itr_count)
            for k in ['loss_bbox', 'loss_ce', 'loss_dice', 'loss_giou', 'loss_mask']:
                writer.add_scalar(k, loss_dict_reduced_scaled[k], itr_count)
            if 'supcon' in loss_dict_reduced_scaled:
                writer.add_scalar('supcon', loss_dict_reduced_scaled['supcon'], itr_count)
        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_ce=loss_dict_reduced_scaled['loss_ce'])
        metric_logger.update(loss_bbox=loss_dict_reduced['loss_bbox'])
        metric_logger.update(loss_giou=loss_dict_reduced['loss_giou'])
        metric_logger.update(dice_loss=loss_dict_reduced_scaled['loss_dice'])
        metric_logger.update(loss_mask=loss_dict_reduced_scaled['loss_mask'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)
        samples, targets = prefetcher.next()
        if debug and j==300:
            break
    torch.cuda.empty_cache()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, args):
    num_frames = args.num_frames
    eval_types = args.eval_types

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco_iou_types = [k for k in ['bbox', 'segm'] if k in postprocessors.keys()]

    coco_evaluator = None
    if 'coco' in eval_types:
        coco_evaluator = CocoEvaluator(base_ds['coco'], coco_iou_types)
        # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    for samples, targets in metric_logger.log_every(data_loader, 1000, header):

        samples = samples.to(device)
        all_outputs, loss_dict = model(samples, targets, criterion, train=False)

        #### reduce losses over all GPUs for logging purposes ####
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        weight_dict = criterion.weight_dict

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        #### reduce losses over all GPUs for logging purposes ####
        ##### single clip input ######

        if all_outputs['pred_boxes'].dim() == 3:
            all_outputs['pred_boxes'] = all_outputs['pred_boxes'].unsqueeze(2)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        results = [{} for i in range(len(targets))]
        if 'bbox' in postprocessors.keys():
            results = postprocessors['bbox'](all_outputs, orig_target_sizes, num_frames=num_frames)

        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, all_outputs, orig_target_sizes, target_sizes)

        res_img = {}

        # evaluate results
        if 'coco' in eval_types:
            for target, output in zip(targets, results):
                for fid in range(num_frames):
                    res_img[target['image_id'][fid].item()] = {}
                    for k, v in output.items():
                        if k == 'masks':
                            res_img[target['image_id'][fid].item()][k] = v[:, fid].unsqueeze(1)
                        elif k == 'boxes':
                            res_img[target['image_id'][fid].item()][k] = v[:, fid]
                        else:
                            res_img[target['image_id'][fid].item()][k] = v

        if coco_evaluator is not None:
            coco_evaluator.update(res_img)

        if args.debug:
            break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    return stats, coco_evaluator

