# ------------------------------------------------------------------------
# InstanceFormer
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import os
import subprocess
import time
from collections import defaultdict, deque
import datetime
import pickle
import random
from typing import Optional, List

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor
from torch.nn.functional import  one_hot
# needed due to empty tensor bug in pytorch and torchvision 0.5
import torchvision
if float(torchvision.__version__[:3]) < 0.5:
    import math
    from torchvision.ops.misc import _NewEmptyTensorOp
    def _check_size_scale_factor(dim, size, scale_factor):
        # type: (int, Optional[List[int]], Optional[float]) -> None
        if size is None and scale_factor is None:
            raise ValueError("either size or scale_factor should be defined")
        if size is not None and scale_factor is not None:
            raise ValueError("only one of size or scale_factor should be defined")
        if not (scale_factor is not None and len(scale_factor) != dim):
            raise ValueError(
                "scale_factor shape must match input shape. "
                "Input is {}D, scale_factor size is {}".format(dim, len(scale_factor))
            )
    def _output_size(dim, input, size, scale_factor):
        # type: (int, Tensor, Optional[List[int]], Optional[float]) -> List[int]
        assert dim == 2
        _check_size_scale_factor(dim, size, scale_factor)
        if size is not None:
            return size
        # if dim is not 2 or scale_factor is iterable use _ntuple instead of concat
        assert scale_factor is not None and isinstance(scale_factor, (int, float))
        scale_factors = [scale_factor, scale_factor]
        # math.floor might return float in py2.7
        return [
            int(math.floor(input.size(i + 2) * scale_factors[i])) for i in range(dim)
        ]
elif float(torchvision.__version__[:3]) < 0.7:
    from torchvision.ops import _new_empty_tensor
    from torchvision.ops.misc import _output_size


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it \n)'.format(
            header, total_time_str, total_time / len(iterable)))


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0], size_divisibility=32)
    return tuple(batch)


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def nested_tensor_from_tensor_list(tensor_list: List[Tensor], size_divisibility=1, split=True, append_instance= False, num_instance = None):
    if split:
        tensor_list = [tensor.split(3,dim=0) for tensor in tensor_list]
        tensor_list = [item for sublist in tensor_list for item in sublist]

    # TODO make this more general
    if tensor_list[0].ndim == 3 and not append_instance: # only for mask
        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])

        if size_divisibility > 1:
            stride = size_divisibility
            # the last two dims are H,W, both subject to divisibility requirement
            max_size[-2] = (max_size[-2] + (stride - 1)) // stride * stride
            max_size[-1] = (max_size[-1] + (stride - 1)) // stride * stride

        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    elif append_instance: #only for masks add instance wise
            # TODO make it support different-sized images
            max_size = _max_by_axis([list(img.shape) for img in tensor_list])

            if size_divisibility > 1:
                stride = size_divisibility
                # the last two dims are H,W, both subject to divisibility requirement
                max_size[-2] = (max_size[-2] + (stride - 1)) // stride * stride
                max_size[-1] = (max_size[-1] + (stride - 1)) // stride * stride

            # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
            max_size[1] = num_instance  #replace total number of instances per batch
            dtype = tensor_list[0].dtype
            device = tensor_list[0].device
            tensor = torch.zeros(max_size, dtype=dtype, device=device)
            start_idx = 0
            for b_img in tensor_list:
                end_idx = start_idx + b_img.shape[1]
                tensor[:,start_idx:end_idx,:b_img.shape[2],:b_img.shape[3]].copy_(b_img)
                start_idx = end_idx
            return tensor,None

            # max_size[0] = len(batch_idx) #replace total number of instances per batch
            # frames, num_ins_in_batch, h, w = max_size # [#frmaes, #num_of_instances,H,W]
            # dtype = tensor_list[0].dtype
            # device = tensor_list[0].device
            # tensor = torch.zeros(max_size, dtype=dtype, device=device)
            # mask = torch.ones((num_ins_in_batch, h, w), dtype=torch.bool, device=device)
            # for bid, i_ind, pad_img, m in zip(batch_idx, mask_ind, tensor, mask): #batch id, instance id in the batch,tensor,mask
            #     img = tensor_list[bid][i_ind, ...]
            #     pad_img[: img.shape[0], : img.shape[1]].copy_(img)
            #     m[: img.shape[0], :img.shape[1]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device, non_blocking=False):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device, non_blocking=non_blocking)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device, non_blocking=non_blocking)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def record_stream(self, *args, **kwargs):
        self.tensors.record_stream(*args, **kwargs)
        if self.mask is not None:
            self.mask.record_stream(*args, **kwargs)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_local_size():
    if not is_dist_avail_and_initialized():
        return 1
    return int(os.environ['LOCAL_SIZE'])


def get_local_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return int(os.environ['LOCAL_RANK'])


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.dist_url = 'env://'
        os.environ['LOCAL_SIZE'] = str(torch.cuda.device_count())
    elif 'SLURM_PROCID' in os.environ:
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput(
            'scontrol show hostname {} | head -n1'.format(node_list))
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
        os.environ['LOCAL_SIZE'] = str(num_gpus)
        args.dist_url = 'env://'
        args.world_size = ntasks
        args.rank = proc_id
        args.gpu = proc_id % num_gpus
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl' if args.debug else 'gloo'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    if float(torchvision.__version__[:3]) < 0.7:
        if input.numel() > 0:
            return torch.nn.functional.interpolate(
                input, size, scale_factor, mode, align_corners
            )

        output_shape = _output_size(2, input, size, scale_factor)
        output_shape = list(input.shape[:-2]) + list(output_shape)
        if float(torchvision.__version__[:3]) < 0.5:
            return _NewEmptyTensorOp.apply(input, output_shape)
        return _new_empty_tensor(input, output_shape)
    else:
        return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)


def get_total_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    device = parameters[0].grad.device
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
                            norm_type)
    return total_norm

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)


def get_top_mem_idx(outputs, mtoken, indices=None, val=False, coco=False, memory = None, pos_query=None,
                    targets = None, pred_boxes =None, nf = None, gt_usage=None):
    if torch.is_tensor(outputs):
        pred_logits = outputs
    else:
        pred_logits = outputs['pred_logits'].squeeze()
    if not val:
        with torch.no_grad():
            batches = []; mem_indices = []; tgt_indices = []; supcon_indices= []
            num_token =  mtoken
            for i, ind in enumerate(indices):
                if isinstance(ind, list):
                    ind = ind[0]
                topk = len(ind[0]) + mtoken
                top_indices = torch.topk(torch.max(pred_logits[i,...].sigmoid(),dim=-1)[0],k=topk)[1] .squeeze().data.cpu()
                top_indices = [a for a in top_indices if a not in ind[0]] # take only new ind
                mem_indices.append(torch.cat((ind[0], torch.stack(top_indices)))[:num_token])
                supcon_indices.append(torch.cat((ind[0], torch.full((max(0, mtoken-len(ind[0])) + mtoken,), 301)))[:num_token]) #todo perhaps better to remove
                batches.append(torch.full((num_token,), i))
                tgt_indices.append(ind[1][:num_token])
            if coco:
                return ([torch.stack(batches),torch.stack(mem_indices)], tgt_indices), torch.stack(supcon_indices)
            elif targets !=None: # for train time dynamic memeory
                top_idx = [torch.stack(batches), torch.stack(mem_indices)]
                top_memory, memory_pos, permute_token = concat_output(memory, pos_query, pred_logits, pred_boxes, top_idx, num_token, gt_usage, targets,
                                                          tgt_indices, nf)
                top_idx[1] = top_idx[1][:, permute_token]
                return (top_memory, memory_pos), torch.stack(supcon_indices)[:, permute_token], top_idx

    else:
        topk = []; batches = []
        for i, pred_cls in enumerate(pred_logits):
            topk.append(torch.topk(torch.max(pred_cls.sigmoid(),dim=-1)[0],k=mtoken)[1])
            batches.append(torch.full((mtoken,), i))
        top_idx = [torch.stack(batches),torch.cat(topk)]
        # todo random permute is ok, but check wd training
        top_memory, memory_pos, permute_token = concat_output(memory, pos_query, pred_logits, pred_boxes, top_idx, mtoken)
        return (top_memory, memory_pos), None, None


def concat_output(mem,mem_pos,cls,box,top_idx, mtoken, gt_usage=None,targets=None, tgt_indices=None,nf=None):
    #with torch.no_grad():
        cls_out = cls[top_idx].sigmoid() #todo do we need sigmoid here? as no gradient
        box_out = box[top_idx]
        #randomly permute the indexes
        permute_token = torch.randperm(mtoken)

        if gt_usage !=None and torch.rand(1) > (1 - gt_usage):  # gradually reduce the gt
            for j, t in enumerate(targets):
                # take only valid class and boxes
                valid_boxes_idx =  t['valid_box'][nf][tgt_indices[j]]
                valid_cls_idx = t['valid_cls'][nf][tgt_indices[j]]

                box_out[j, :len(tgt_indices[j]), ...][valid_boxes_idx] = t['boxes'][nf, tgt_indices[j],...][valid_boxes_idx]
                cls_out[j, :len(tgt_indices[j]), ...][valid_cls_idx] = one_hot(t['labels'][nf][tgt_indices[j]][valid_cls_idx],cls.shape[-1]).type(cls_out.dtype)  # make one hot number of class

        return torch.cat((mem[top_idx], cls_out, box_out), dim=-1)[:,permute_token, ...], mem_pos[top_idx][:,permute_token, ...],permute_token  # randomly permute


def valid_objs(self, target):
    labels = target['labels'].repeat(self.num_frames, 1)
    target['occl'] = torch.zeros(1, dtype=torch.bool)
    target["valid_box"] = torch.ones(labels.shape, dtype=torch.bool)
    target['valid_cls'] = target["valid_box"].clone()

    target['matcher_label'] = target['labels'].repeat(self.num_frames,1) # only for matcher as max_class-1
    target['labels'] = labels
    return  target