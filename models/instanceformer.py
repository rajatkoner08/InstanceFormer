# ------------------------------------------------------------------------
# InstanceFormer model and criterion classes.
# ------------------------------------------------------------------------
# Modified from SeqFormer (https://github.com/wjf5203/SeqFormer)
# Copyright (c) 2021 Junfeng Wu. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


"""
Deformable DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import math
from .position_encoding import PositionEmbeddingSine
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list, get_world_size, is_dist_avail_and_initialized, inverse_sigmoid)
from .backbone import build_backbone
from .matcher import build_matcher
from .online_segmentation import (InstanceFormer,PostProcessSegm,dice_loss, sigmoid_focal_loss)
from .deformable_transformer import build_deforamble_transformer
import copy
from util.supcon import SupConLoss


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """

    def __init__(self, backbone, transformer, num_classes, num_frames, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, propagate_class=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
        """
        super().__init__()
        self.num_frames = num_frames
        self.num_queries = num_queries
        self.transformer = transformer
        self.num_classes = num_classes
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.propagate_class = propagate_class
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels
        self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        self.mtoken = transformer.mtoken
        self.mframe =  transformer.mframe
        if self.mtoken > 0 and self.mframe > 0: # for temporal and spatial memory support
            self.memory_pos = PositionEmbeddingSine(num_pos_feats=(hidden_dim)//2,normalize=True)
        if self.propagate_class:
            self.temporal_class_embed = nn.Linear(num_classes, num_classes) # temporal either make it 1 or num class for exp
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        # if self.propagate_class:
        #     self.temporal_class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            if self.propagate_class:
                self.temporal_class_embed = _get_clones(self.temporal_class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            if self.propagate_class:
                self.temporal_class_embed = nn.ModuleList([self.temporal_class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [num_frames x 3 x H x W]
               - samples.mask: a binary mask of shape [num_frames x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """

        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)

        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        poses = []
        for l, feat in enumerate(features[1:]):
            # src: [nf*N, _C, Hi, Wi],
            # mask: [nf*N, Hi, Wi],
            # pos: [nf*N, C, H_p, W_p]
            src, mask = feat.decompose()
            src_proj_l = self.input_proj[l](src)  # src_proj_l: [nf*N, C, Hi, Wi]

            # src_proj_l -> [nf, N, C, Hi, Wi]
            n, c, h, w = src_proj_l.shape
            src_proj_l = src_proj_l.reshape(n // self.num_frames, self.num_frames, c, h, w).permute(1, 0, 2, 3, 4)

            # mask -> [nf, N, Hi, Wi]
            mask = mask.reshape(n // self.num_frames, self.num_frames, h, w).permute(1, 0, 2, 3)

            # pos -> [nf, N, Hi, Wi]
            np, cp, hp, wp = pos[l + 1].shape
            pos_l = pos[l + 1].reshape(np // self.num_frames, self.num_frames, cp, hp, wp).permute(1, 0, 2, 3, 4)
            for n_f in range(self.num_frames):
                srcs.append(src_proj_l[n_f])
                masks.append(mask[n_f])
                poses.append(pos_l[n_f])
                assert mask is not None

        if self.num_feature_levels > (len(features) - 1):
            _len_srcs = len(features) - 1
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask  # [nf*N, H, W]
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)

                # src -> [nf, N, C, H, W]
                n, c, h, w = src.shape
                src = src.reshape(n // self.num_frames, self.num_frames, c, h, w).permute(1, 0, 2, 3, 4)
                mask = mask.reshape(n // self.num_frames, self.num_frames, h, w).permute(1, 0, 2, 3)
                np, cp, hp, wp = pos_l.shape
                pos_l = pos_l.reshape(np // self.num_frames, self.num_frames, cp, hp, wp).permute(1, 0, 2, 3, 4)

                for n_f in range(self.num_frames):
                    srcs.append(src[n_f])
                    masks.append(mask[n_f])
                    poses.append(pos_l[n_f])

        query_embeds = None
        query_embeds = self.query_embed.weight
        hs, memory, init_reference, inter_references,\
                    enc_inter_outputs_class, enc_inter_outputs_coord = self.transformer(srcs, masks, poses, query_embeds)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25, mask_out_stride=4, num_frames=1, batch_size=2, supcon=False):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.mask_out_stride = mask_out_stride
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.supcon = SupConLoss() if supcon else None

    def loss_labels(self, outputs, targets, indices, num_classes, log=True, **kwargs):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        num_frames = self.num_frames -1 if 'aux_outputs_enc' in kwargs.keys() else self.num_frames
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        loss_ce = 0
        #for i, indices in enumerate(indices_list): #loop over frame and indices

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][0][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[1:3], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o # instance sud remain in same pos for every frame

        target_classes_onehot = torch.zeros([src_logits.shape[1], src_logits.shape[2], src_logits.shape[3] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        for i in range(num_frames):
            loss_ce = loss_ce + sigmoid_focal_loss(src_logits[i,...], target_classes_onehot, num_classes[i], alpha=self.focal_alpha, gamma=2) * \
                          src_logits.shape[2]

        losses = {'loss_ce': loss_ce/num_frames}

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes, **kwargs):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        num_frames = self.num_frames -1 if 'aux_outputs_enc' in kwargs.keys() else self.num_frames
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"][0]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(2)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.repeat(num_frames,1).float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes, **kwargs):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        num_frames = self.num_frames -1 if 'aux_outputs_enc' in kwargs.keys() else self.num_frames

        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)

        if 'aux_outputs_enc' in kwargs.keys():
            assert num_frames == outputs['pred_boxes'].shape[0] == targets[0]['boxes'].shape[0] -1
            tgt_boxes = torch.cat([v["boxes"][1:] for v in targets], 1)
        else:
            assert num_frames==outputs['pred_boxes'].shape[0]==targets[0]['boxes'].shape[0]
            tgt_boxes = torch.cat([v["boxes"] for v in targets],1)
            valid_boxes = tgt_boxes # torch.cat([v["valid_box"] for v in targets],1) # skips losses for boxes if instances are occluded,not yet bron or dead

        loss_giou = 0
        loss_bbox = 0
        for i, (frame_boxes,tgt_bbox, v_boxes) in enumerate(zip(outputs['pred_boxes'],tgt_boxes,valid_boxes)):
            #idx, order_batch_idx = self._get_src_permutation_idx(indices, order_batch=True)
            src_boxes = frame_boxes[idx] #[v_boxes[order_batch_idx],:]  # with only valid boxes

            sizes = [len(v["labels"][i]) for v in targets]
            tgt_bbox = list(tgt_bbox.split(sizes, dim=0))

            tgt_bbox = torch.cat([t[i] for t, (_, i) in zip(tgt_bbox, indices)], dim=0)
            #tgt_bbox = tgt_bbox[v_boxes[order_batch_idx],:]
            loss_bbox = loss_bbox + F.l1_loss(src_boxes.flatten(0, 1), tgt_bbox.flatten(0, 1), reduction='none')

            loss_giou = loss_giou + 1 - torch.diag(box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes),
                box_ops.box_cxcywh_to_xyxy(tgt_bbox)))

        loss_bbox = loss_bbox / num_frames
        loss_giou = loss_giou / num_frames

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes[0]
        losses['loss_giou'] = loss_giou.sum() / num_boxes[0]
        #     loss_bbox = loss_bbox + F.l1_loss(src_boxes.flatten(0, 1), tgt_bbox.flatten(0,1), reduction='none')#/num_boxes[i]
        #
        #     loss_giou = loss_giou + sum(1 - torch.diag(box_ops.generalized_box_iou(
        #             box_ops.box_cxcywh_to_xyxy(src_boxes),
        #             box_ops.box_cxcywh_to_xyxy(tgt_bbox))))/num_boxes[i]
        #
        #
        # losses = {}
        # losses['loss_bbox'] = loss_bbox / num_frames
        # losses['loss_giou'] = loss_giou / num_frames

        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes, **kwargs):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs
        tgt_idx = self._get_tgt_permutation_idx(indices)
        tgt_idx = self._get_tgt_instance_idx(tgt_idx)
        if kwargs['target_masks'] ==None:
            gt_masks,_ = nested_tensor_from_tensor_list([t["masks"] for t in targets],  # iterate frame wise
                                           size_divisibility=32, split=False, append_instance=True, num_instance = sum([len(ind_[1]) for ind_ in indices]))
            gt_masks = gt_masks.to(outputs["pred_masks"])
        else:
            gt_masks = kwargs['target_masks']

        valid_boxes = gt_masks #torch.cat([v["valid_box"] for v in targets],1)  # skips losses for mask if instances are occluded,not yet bron or dead

        loss_mask = 0
        loss_dice = 0
        for i, (src_masks,target_masks, v_boxes) in enumerate(zip(outputs["pred_masks"],gt_masks, valid_boxes)): #iterate frame wise
            if type(src_masks) == list:
                src_masks = torch.cat(src_masks, dim=1)[0]

            # downsample ground truth masks with ratio mask_out_stride
            start = int(self.mask_out_stride // 2)
            im_h, im_w = target_masks.shape[-2:]
            target_masks = target_masks[ :, start::self.mask_out_stride, start::self.mask_out_stride]
            assert target_masks.size(1) * self.mask_out_stride == im_h
            assert target_masks.size(2) * self.mask_out_stride == im_w

            #  # take only valid masks

            src_masks = src_masks.flatten(1) #src_masks[v_boxes[tgt_idx[1]]].flatten(1)
            target_masks = target_masks[tgt_idx[1]].flatten(1) # #todo check this logic --checked..sud be ok now

            #target_masks = target_masks[v_boxes[tgt_idx[1]]].flatten(1)

            # src_masks/target_masks: [n_targets, H * W]
            loss_mask = loss_mask+ sigmoid_focal_loss(src_masks, target_masks, num_boxes[i])
            loss_dice = loss_dice+ dice_loss(src_masks, target_masks, num_boxes[i])
        losses = {
            "loss_mask": loss_mask/self.num_frames,
            "loss_dice": loss_dice/self.num_frames,
        }
        return losses

    def _get_src_permutation_idx(self, indices,order_batch=False):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        if order_batch:
            b_len =  [len(src) for (src,_) in indices]
            ordered_batch_idx = torch.cat([idx + sum(b_len[:i]) for i, (_, idx) in enumerate(indices)])
            return (batch_idx, src_idx), ordered_batch_idx
        else:
            return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx


    def _get_tgt_instance_idx(self, tgt_idx):
        # get batch wise target instances
        previous_b = 0
        batch_ofset = 0
        for i, (b, id) in enumerate(zip(tgt_idx[0], tgt_idx[1])):
            if b != previous_b:
                batch_ofset = i
                previous_b = b
            tgt_idx[1][i] = id + batch_ofset  #
        return tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, indices_list, valid_ratios):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Compute the average number of target boxes accross all nodes, for normalization purposes

        # take framewise valid class and boxes
        num_classes = torch.sum(torch.cat(([t["valid_cls"] for t in targets]), dim =-1),axis=-1).to(next(iter(outputs.values())).device)
        #num_boxes = torch.sum(torch.cat(([t["valid_box"] for t in targets]), dim =-1),axis=-1).to(next(iter(outputs.values())).device)
        # for i,(nb,nc) in enumerate(zip(num_boxes,num_classes)):
        #     if nb!=nc:
        #         print('test')
        # #num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            #torch.distributed.all_reduce(num_boxes)
            torch.distributed.all_reduce(num_classes)
        #num_boxes = torch.clamp(num_boxes / get_world_size(), min=1)
        num_boxes = num_classes = torch.clamp(num_classes / get_world_size(), min=1)

        # compute target masks from nested tensor list only once here
        target_masks, _ = nested_tensor_from_tensor_list([t["masks"] for t in targets],  # iterate frame wise
                                                     size_divisibility=32, split=False, append_instance=True,
                                                     num_instance=sum([len(ind_[1]) for ind_ in indices_list[0]]))
        target_masks = target_masks.to(outputs["pred_masks"])

        # Compute all the requested losses
        losses = {}
        #indices_list = [indices_list]*self.num_frames #repeat the list for all num of frame
        for loss in self.losses:
            kwargs = {}
            kwargs['target_masks'] = target_masks
            losses.update(self.get_loss(loss, outputs, targets, indices_list[-1],num_classes if loss=='labels' else num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):

                for loss in self.losses:
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    #     continue
                    kwargs = {}
                    kwargs['target_masks'] = target_masks
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_list[i], num_classes if loss=='labels' else num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs: # todo losses are not implements as per dynamic class and boxes
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        if 'aux_outputs_enc' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs_enc']):
                nf = targets[0]['boxes'].shape[0]-1
                indices = self.matcher(aux_outputs, targets, nf, None, False, True)
                aux_outputs['pred_logits'] = aux_outputs['pred_logits'].transpose(1, 0)
                aux_outputs['pred_boxes'] = aux_outputs['pred_boxes'].transpose(1, 0)
                for loss in self.losses:
                    if loss in ['masks', "mask_prediction", "corr"]:
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    kwargs['aux_outputs_enc'] = True
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if self.supcon is not None:
            assert 'supcon_idx' in outputs
            loss = 0
            for i, (supcon_feat, supcon_index) in enumerate(zip(outputs['supcon_feats'],outputs['supcon_idx'])): #loop over batch
                supcon_feat =  F.normalize(supcon_feat, p=2, dim=2)
                loss+=self.supcon(features=supcon_feat.flatten(0,1)[:,None,:],
                                  labels=supcon_index.flatten()[:, None])
            losses.update({'supcon': loss/(supcon_feat.shape[0]*supcon_feat.shape[1])})

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes, num_frames=1):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
            num_frames: output frame num
        """
        # output single / multi frames
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        # out_logits: [N, num_queries, num_classes]
        # out_bbox: [N, num_queries, num_frames, 4]

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        bs, num_q = out_logits.shape[:2]

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]

        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, boxes.shape[-2], boxes.shape[-1]))
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, None, :]
        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    args.num_classes = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        args.num_classes = 250
    if 'ovis' in args.dataset_file:
        args.num_classes = 27
    if args.dataset_file == 'YoutubeVIS' or args.dataset_file == 'jointcoco' or args.dataset_file == 'Seq_coco':
        args.num_classes = 42

    device = torch.device(args.device)

    if 'swin' in args.backbone:
        from .swin_transformer import build_swin_backbone
        backbone = build_swin_backbone(args)
    else:
        backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args)
    model = DeformableDETR(
        backbone,
        transformer,
        num_classes=args.num_classes,
        num_frames=1 ,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        propagate_class = args.propagate_class,
    )
    if args.masks:
        model = InstanceFormer(model, num_frames=args.num_frames,
                               rel_coord=args.rel_coord, propagate_ref_points=args.propagate_ref_points,
                               propagate_class=args.propagate_class,
                               supcon=args.supcon, analysis=args.analysis)

    matcher = build_matcher(args)

    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    if args.supcon:
        weight_dict.update({'supcon': 2})

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]

    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = SetCriterion(args.num_classes, matcher, weight_dict, losses,
                             mask_out_stride=args.mask_out_stride,
                             focal_alpha=args.focal_alpha,
                             num_frames= args.num_frames, batch_size=args.batch_size,
                             supcon=args.supcon)
    criterion.to(device)

    postprocessors = {'bbox': PostProcess()}
    # postprocessors = {}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()

    return model, criterion, postprocessors
