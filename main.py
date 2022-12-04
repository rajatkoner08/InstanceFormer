# ------------------------------------------------------------------------
# Training script of InstanceFormer
# ------------------------------------------------------------------------
# Modified from Deformable DETR and SeqFormer
# ------------------------------------------------------------------------


import argparse
import datetime
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets
from torch.utils.tensorboard import SummaryWriter
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model

from arg_parse import get_args_parser


def main(args):
    utils.init_distributed_mode(args)
    args.train = True
    device = torch.device(args.device)
    if args.debug:  # only for debug
        args.num_workers = 0

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
            sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        # sampler_val = torch.utils.data.SequentialSampler(dataset_val, shuffle=False)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                   pin_memory=True)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                 pin_memory=True)

    # lr_backbone_names = ["backbone.0", "backbone.neck", "input_proj", "transformer.encoder"]
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n,
                                                                                                   args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if
                       match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if
                       match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]

    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_drop)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    base_ds = {}

    if args.dataset_file == 'YoutubeVIS' or 'joint' in args.dataset_file or args.dataset_file == 'Seq_coco':
        base_ds['ytvos'] = get_coco_api_from_dataset(dataset_val)
    else:
        base_ds['coco'] = get_coco_api_from_dataset(dataset_val)

    tensorboard_step = 0

    if args.resume:# is not None:
        resume_path = f'{args.initial_output_dir}/experiments/{args.exp_name}/checkpoint.pth'
        print('resume from ', resume_path)
        checkpoint = torch.load(resume_path, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=True)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            # print(optimizer.param_groups)
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                print(
                    'Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.last_epoch = args.start_epoch
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1

    elif args.pretrain_weights is not None:
        print('load weigth from pretrain weight:', args.pretrain_weights)
        checkpoint = torch.load(args.pretrain_weights, map_location='cpu')['model']
        if not args.jointfinetune:
            print('delete all class embedding ')
            del_list = []
            for k, v in checkpoint.copy().items():
                if 'deformable_detr' in args.pretrain_weights:  # in case if we are loading weight of deformable deter
                    if 'transformer.' in k:
                        checkpoint['detr.' + k] = v
                        del_list.append(k)
                if 'detr.class_embed' in k:
                    del_list.append(k)
                if args.skip_pretrain_detr_decoder:
                    if 'detr.transformer.decoder' in k:
                        del_list.append(k)
                if args.skip_pretrain_detr_temporal_class:
                    if 'detr.temporal_class_embed' in k:
                        del_list.append(k)
            for k in del_list:
                del checkpoint[k]
            model_without_ddp.load_state_dict(checkpoint, strict=False)
        else:
            print(' keep all the weights')
            del_list = []
            if args.skip_pretrain_detr_decoder:
                for k, v in checkpoint.copy().items():
                    if 'detr.transformer.decoder' in k:
                            del_list.append(k)
                for k in del_list:
                    del checkpoint[k]
            model_without_ddp.load_state_dict(checkpoint, strict=True)

    output_dir = os.path.join(args.initial_output_dir, "experiments", args.exp_name)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(os.path.join(output_dir, 'config.txt'), 'w') as convert_file:
        convert_file.write(json.dumps(vars(args)))
    if args.writer:
        writer = SummaryWriter(log_dir=os.path.join(args.initial_output_dir, "tensorboard", '%s_%d' % (args.exp_name, args.seed)))
    else:
        writer = None

    print("Start training")
    start_time = time.time()

    gt_usage = [0.7,0.5,0.3,0.15,0.05] # amount of gt data to be used

    for epoch in range(args.start_epoch, args.epochs):
        model.gt_usage = gt_usage[-1] if epoch >= len(gt_usage) else gt_usage[epoch]
        if args.distributed:
            sampler_train.set_epoch(epoch)
        #current number of frames
        if args.increase_nf_epochs !=None and  epoch in args.increase_nf_epochs:
            curr_nf = args.num_frames + args.increase_nf_epochs.index(epoch)+1
            model.num_frames = curr_nf
            criterion.num_frames = curr_nf
            if args.distributed:
                model.module.num_frames = curr_nf
            if 'joint' in args.dataset_file:
                data_loader_train.dataset.datasets[0].num_frames = curr_nf
                data_loader_train.dataset.datasets[1].num_frames = curr_nf
            else:
                data_loader_train.dataset.num_frames = curr_nf
            print("\n############## Increased Number of Frame to ",curr_nf," ###################\n")

        epoch_start_time = time.time()
        train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm,
                                      writer, args.debug, args.frame_wise_loss)
        epoch_total_time = time.time() - epoch_start_time
        epoch_time_str = str(datetime.timedelta(seconds=int(epoch_total_time)))
        print('Total training time {} for epoch{} '.format(epoch_time_str,epoch))

        lr_scheduler.step()
        if args.exp_name:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 2 epochs
            if (epoch + 1) % 1 == 0 or (epoch + 1) % 1 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'tensorboard_step': tensorboard_step
                }, checkpoint_path)

        if (epoch + 1) % 1 == 0 and args.eval_types == 'coco':
            test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                                  data_loader_val, base_ds, device, args)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

            if args.exp_name and utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                if coco_evaluator is not None:
                    (output_dir / 'eval').mkdir(exist_ok=True)
                    if "segm" in coco_evaluator.coco_eval:
                        filenames = ['coco_latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'coco_{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["segm"].eval,
                                       output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('InstanceFormer training script', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.cuda_visible_devices))

    main(args)



