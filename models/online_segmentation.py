# ------------------------------------------------------------------------
# InstanceFormer Transformer classes.
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
This file provides the definition of the convolutional heads used to predict masks, as well as the losses
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.segmentation_helper import run_backbone,transformer_postprocess,get_memories
from util.misc import NestedTensor, nested_tensor_from_tensor_list


class InstanceFormer(nn.Module):
    def __init__(self, detr, rel_coord=True, num_frames=1, propagate_ref_points=False,
                 propagate_class=False, supcon=False, analysis=False):
        super().__init__()
        self.detr = detr
        self.rel_coord = rel_coord
        self.num_frames = num_frames
        self.propagate_ref_points = propagate_ref_points
        self.propagate_class = propagate_class
        self.supcon = supcon
        self.gt_usage = 0
        hidden_dim, nheads = detr.transformer.d_model, detr.transformer.nhead
        self.in_channels = hidden_dim // 32
        self.dynamic_mask_channels = 8
        self.controller_layers = 3
        self.max_insts_num = 100
        self.mask_out_stride = 4
        self.analysis = analysis

        # dynamic_mask_head params
        weight_nums, bias_nums = [], []
        for l in range(self.controller_layers):
            if l == 0:
                if self.rel_coord:
                    weight_nums.append((self.in_channels + 2) * self.dynamic_mask_channels)
                else:
                    weight_nums.append(self.in_channels * self.dynamic_mask_channels)
                bias_nums.append(self.dynamic_mask_channels)
            elif l == self.controller_layers - 1:
                weight_nums.append(self.dynamic_mask_channels * 1)
                bias_nums.append(1)
            else:
                weight_nums.append(self.dynamic_mask_channels * self.dynamic_mask_channels)
                bias_nums.append(self.dynamic_mask_channels)

        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)
        self.controller = MLP(hidden_dim, hidden_dim, self.num_gen_params, 3)
        for contr in self.controller.layers:
            nn.init.xavier_uniform_(contr.weight)
            nn.init.zeros_(contr.bias)

        self.mask_head = MaskHeadSmallConv(hidden_dim, None, hidden_dim)


    def forward(self, samples: NestedTensor, gt_targets, criterion):
        if gt_targets[0]['boxes'].dim() == 2:
            for gt_target in gt_targets:
                gt_target['boxes'] = gt_target['boxes'][None,...]
                gt_target['masks'] = gt_target['masks'][None, ...]
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples, size_divisibility=32)
        #run backbone
        srcs, masks, poses, spatial_shapes, np, c=  run_backbone(self, samples)

        query_embeds = self.detr.query_embed.weight
        # split query  to tgt and position emb
        query_embed, tgt = torch.split(query_embeds, c, dim=1)
        query_embed = query_embed.unsqueeze(0).expand(np // self.num_frames, -1, -1)
        tgt = tgt.unsqueeze(0).expand(np // self.num_frames, -1, -1)

        #load the memeory position encoding
        if self.detr.mframe>0 and self.detr.mtoken:
            #nf,c,b,memory_token
            temporal_mem_pos = self.detr.memory_pos(mask=torch.zeros((np // self.num_frames, self.num_frames,1 ), dtype=torch.bool, device=tgt.device))

        hs_mem = []
        supcon_feats = []
        supcon_idx = []
        hs_mem_pos = []
        all_indices = []
        outputs = {}
        outputs_classes = []
        outputs_coords = []
        outputs_mask = []

        hs = None  # set initial hidden state to None, that it be uses in loop
        init_reference = None
        t_classes = []

        instance_mem = None; instance_mem_pos = None; frame_mem_mask = None; enc_sample_loc =None;
        for frame in range(self.num_frames): #overall dimension [#bs,#frame,#chanel,H,W ]
            frame_lbl_src = [src[:, frame, :, :, :].unsqueeze(1) for src in srcs]
            frame_lbl_mask = [mask[:, frame, :, :].unsqueeze(1) for mask in masks]
            frame_lbl_pos = [pos[:, frame, :, :, :].unsqueeze(1) for pos in poses]

            if frame > 0 and self.detr.mframe > 0 and self.detr.mtoken>0:
                instance_mem, instance_mem_pos, frame_mem_mask = get_memories(self, frame, hs_mem, temporal_mem_pos,hs_mem_pos)


            hs, memory, init_reference, inter_references, inter_samples, enc_outputs_class, valid_ratios,\
            lvl_start_idx, sampling_locations_all = self.detr.transformer(frame_lbl_src, frame_lbl_mask, frame_lbl_pos, query_embed,
                                      tgt if hs==None else hs[-1,...],
                                      (init_reference) if
                                      (self.propagate_ref_points and init_reference is not None) else None,
                                      instance_mem, instance_mem_pos,frame_mem_mask)
            hs = hs.squeeze(2)
            valid_ratios = valid_ratios[:, 0]
            # calculate class , box, seg mask
            o_classes, o_cord, o_maks, top_mem, t_classes, supcon_ind, all_indices, idx_permute = transformer_postprocess(self, hs,memory, init_reference,
                                                                                    inter_references,spatial_shapes,frame, gt_targets = gt_targets,
                                                                                    criterion=criterion,indices_list=all_indices,
                                                                                    t_classes=t_classes, query_embed = query_embed)
            #all_indices.append(frame_indices)
            outputs_classes.append(o_classes)
            outputs_coords.append(o_cord)
            outputs_mask.append(o_maks)

            if self.detr.mframe >0  and self.detr.mtoken > 0:
                #sud nt be any gradient
                hs_mem.append(top_mem[0])
                supcon_feats.append(hs[-1,...][idx_permute])
                hs_mem_pos.append(top_mem[1])
                supcon_idx.append(supcon_ind)

        # bs, outputs_mask = len(outputs_masks[0]), []
        # outputs_masks: dec_num x bs x [1, num_insts, 1, h, w]
        outputs_class = torch.stack(outputs_classes) #[#frames,#layer,bs,#query,#numclasses]
        outputs_coord = torch.stack(outputs_coords) #[#frames,#layer,bs,#query,#numclasses]
        outputs_mask = torch.stack(outputs_mask) #frames,#layer,#num instances,1,H,W]
        # outputs['pred_samples'] = inter_samples[-1]
        outputs['pred_logits'] = outputs_class[:,-1,...] #[#frames,bs,#query,#numclasses]
        outputs['pred_boxes'] = outputs_coord[:,-1,...]
        outputs['pred_masks'] = outputs_mask[:,-1,...] # [masks[-1] for masks in outputs_mask]

        if self.detr.aux_loss:
            outputs['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_mask)
        if self.supcon:
            outputs['supcon_feats'] = torch.stack(supcon_feats).transpose(1,0) #b,nf,mtoken,hidden_dim
            outputs['supcon_idx'] = torch.stack(supcon_idx).transpose(1,0)

        loss_dict = criterion(outputs, gt_targets, all_indices, valid_ratios)

        return outputs, loss_dict


    def inference(self, samples: NestedTensor, orig_w, orig_h, temporal_mem_pos, fast_inference):
        if fast_inference:
            if not isinstance(samples, NestedTensor):
                samples = nested_tensor_from_tensor_list(samples)
            #run backbone
            srcs, masks, poses, spatial_shapes, np, c = run_backbone(self, samples, train=False)

        c = [a for a in self.detr.input_proj.parameters()][-1].shape[0]
        #query
        query_embeds = self.detr.query_embed.weight
        # split query  to tgt and position emb
        query_embed, tgt = torch.split(query_embeds, c, dim=1)
        query_embed = query_embed.unsqueeze(0)# .expand(np // self.num_frames, -1, -1)
        tgt = tgt.unsqueeze(0)#.expand(np // self.num_frames, -1, -1)

        #nf,c,b,memory_token
        if self.detr.mframe > 0 and self.detr.mtoken >0 :
            temporal_mem_pos = self.detr.memory_pos(mask=torch.zeros((1, self.detr.mframe - 1, 1), dtype=torch.bool, device=tgt.device)) # for inference bs=1

        outputs = {}
        outputs_classes = []
        outputs_coords = []
        outputs_mask = []
        outputs_features = []
        hs_mem = []
        hs_mem_pos = []
        hs = None  # set initial hidden state to None, that it be uses in loop
        init_reference = None
        t_classes = []
        frame_mem = frame_mem_pos = frame_mem_mask = None;


        for frame in range(self.num_frames):  # overall dimension [#bs,#frame,#chanel,H,W ]
            if fast_inference:
                frame_lbl_src = [src[frame, :, :, :][None, None, ...] for src in srcs]
                frame_lbl_mask = [mask[frame, :, :][None, None, ...] for mask in masks]
                frame_lbl_pos = [pos[frame, :, :, :][None, None, ...] for pos in poses]
            else:
                # run backbone for image
                sample = nested_tensor_from_tensor_list(samples[frame,None])
                srcs, masks, poses, spatial_shapes, np, c = run_backbone(self, sample, train=False)

                frame_lbl_src = [src[0, :, :, :][None,None,...] for src in srcs]
                frame_lbl_mask = [mask[0, :, :][None,None,...] for mask in masks]
                frame_lbl_pos = [pos[0, :, :, :][None,None,...] for pos in poses]

            #add st pos embedding with mem , ready for transformer
            if self.detr.mframe > 0 and self.detr.mtoken > 0 and frame >0:
                frame_mem, frame_mem_pos, frame_mem_mask = get_memories(self, frame, hs_mem, temporal_mem_pos, hs_mem_pos=hs_mem_pos)

            #call transformer for each frame
            hs, memory, init_reference, inter_references, inter_samples, enc_outputs_class, \
            valid_ratios, lvl_start_idx, sampling_locations_all = self.detr.transformer(frame_lbl_src, frame_lbl_mask, frame_lbl_pos,
                query_embed, tgt if hs == None else hs[-1, ...], (init_reference) if self.propagate_ref_points else None,
                frame_mem, frame_mem_pos, frame_mem_mask)

            hs = hs.squeeze(2)
            # calculate class, box, mask
            o_classes, o_cord, o_maks, top_mem, t_classes, _, _, _ = transformer_postprocess(self, hs, memory, init_reference,
                                                                                       inter_references,
                                                                                       spatial_shapes, frame, orig_w,
                                                                                       orig_h, t_classes=t_classes,
                                                                                       final_only=True,
                                                                                       query_embed=query_embed)

            outputs_classes.append(o_classes)
            outputs_coords.append(o_cord)
            outputs_mask.append(o_maks)
            outputs_features.append(hs[-1]) #MODIF

            if self.detr.mframe > 0 and self.detr.mtoken > 0:
                hs_mem.append(top_mem[0])
                hs_mem_pos.append(top_mem[1])
                if len(hs_mem) > self.detr.mframe-1: # delete old memories
                    del hs_mem[0]
                    del hs_mem_pos[0]

            if self.analysis:
                if (frame==0):
                    num_queries_, query_dim_ = hs[-1, ...].squeeze().size()
                    layers_,_, heads, scales, num_samp_, num_coord = sampling_locations_all.squeeze()[..., :2].size()
                    hs_analysis = torch.empty((self.num_frames,num_queries_,query_dim_ )).to(query_embed)
                    ref_analysis = torch.empty( (self.num_frames,1,num_queries_, heads, scales, num_samp_, num_coord) ).to(query_embed)

                    init_ref_analysis= torch.empty( (self.num_frames, num_queries_, num_coord) ).to(query_embed)

                hs_analysis[frame] = hs[-1, ...].squeeze()
                ref_analysis[frame] = sampling_locations_all.squeeze()[...,:2].sigmoid()[-1,...].unsqueeze(0)
                init_ref_analysis[frame] = init_reference.squeeze()

        outputs_class = torch.stack(outputs_classes)  # [#frames,#layer,bs,#query,#numclasses]
        outputs_coord = torch.stack(outputs_coords)  # [#frames,#layer,bs,#query,#numclasses]
        outputs_mask = torch.stack(outputs_mask)  # frames,#layer,#num instances,1,H,W]
        outputs_features = torch.stack(outputs_features) #MODIF
        outputs['pred_logits'] = outputs_class[:, -1, ...]  # [#frames,bs,#query,#numclasses]
        outputs['pred_boxes'] = outputs_coord[:, -1, ...]
        outputs['pred_masks'] = outputs_mask[:, -1, ...]  # [masks[-1] for masks in outputs_mask]
        outputs['pred_features'] = outputs_features.squeeze(1) # MODIF
        torch.cuda.empty_cache()
        if self.analysis:
            outputs['hs_analysis'] = hs_analysis
            outputs['ref_analysis'] = ref_analysis
            outputs['init_ref_analysis'] = init_ref_analysis
        return outputs


    def forward_mask_head_train(self, outputs, feats, spatial_shapes, reference_points, mask_head_params, num_insts):
        bs, n_f, _, c = feats.shape
        # nq = mask_head_params.shape[1]

        # encod_feat_l: num_layers x [bs, C, num_frames, hi, wi]
        encod_feat_l = []
        spatial_indx = 0
        for feat_l in range(self.detr.num_feature_levels - 1):
            h, w = spatial_shapes[feat_l]
            mem_l = feats[:, :, spatial_indx: spatial_indx + h * w, :].reshape(bs, self.detr.num_frames, h, w,
                                                                               c).permute(0, 4, 1, 2, 3)
            encod_feat_l.append(mem_l)
            spatial_indx += h * w
        pred_masks = []
        for iframe in range(self.detr.num_frames):
            encod_feat_f = []
            for lvl in range(self.detr.num_feature_levels - 1):
                encod_feat_f.append(encod_feat_l[lvl][:, :, iframe, :, :])  # [bs, C, hi, wi]

            decod_feat_f = self.mask_head(encod_feat_f, fpns=None)
            # decod_feat_f = self.spatial_decoder(encod_feat_f)[0]
            # [bs, C/32, H/8, W/8]
            reference_points_i = reference_points[:, iframe]
            ######### conv ##########
            mask_logits = self.dynamic_mask_with_coords(decod_feat_f, reference_points_i, mask_head_params,
                                                        num_insts=num_insts,
                                                        mask_feat_stride=8,
                                                        rel_coord=self.rel_coord)
            # mask_logits: [1, num_queries_all, H/4, W/4]

            # mask_f = mask_logits.unsqueeze(2).reshape(bs, nq, 1, decod_feat_f.shape[-2], decod_feat_f.shape[-1])  # [bs, selected_queries, 1, H/4, W/4]
            mask_f = []
            inst_st = 0
            for num_inst in num_insts:
                # [1, selected_queries, 1, H/4, W/4]
                mask_f.append(mask_logits[:, inst_st: inst_st + num_inst, :, :].unsqueeze(2))
                inst_st += num_inst

            pred_masks.append(mask_f)

            # outputs['pred_masks'] = torch.cat(pred_masks, 2) # [bs, selected_queries, num_frames, H/4, W/4]
        output_pred_masks = []
        for i, num_inst in enumerate(num_insts):
            out_masks_b = [m[i] for m in pred_masks]
            output_pred_masks.append(torch.cat(out_masks_b, dim=2))

        outputs['pred_masks'] = output_pred_masks
        return outputs


    def mask_heads_forward(self, features, weights, biases, num_insts):
        '''
        :param features
        :param weights: [w0, w1, ...]
        :param bias: [b0, b1, ...]
        :return:
        '''
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    def dynamic_mask_with_coords(self, mask_feats, reference_points, mask_head_params, num_insts,
                                 mask_feat_stride, rel_coord=True):
        # mask_feats: [N, C/32, H/8, W/8]
        # reference_points: [1, \sum{selected_insts}, 2]
        # mask_head_params: [1, \sum{selected_insts}, num_params]
        # return:
        #     mask_logits: [1, \sum{num_queries}, H/8, W/8]
        device = mask_feats.device

        N, in_channels, H, W = mask_feats.size()
        num_insts_all = reference_points.shape[1]

        locations = compute_locations(
            mask_feats.size(2), mask_feats.size(3),
            device=device, stride=mask_feat_stride)
        # locations: [H*W, 2]

        if rel_coord:
            instance_locations = reference_points
            relative_coords = instance_locations.reshape(1, num_insts_all, 1, 1, 2) - locations.reshape(1, 1, H, W, 2)
            relative_coords = relative_coords.float()
            relative_coords = relative_coords.permute(0, 1, 4, 2, 3).flatten(-2, -1)
            mask_head_inputs = []
            inst_st = 0
            for i, num_inst in enumerate(num_insts):
                # [1, num_queries * (C/32+2), H/8 * W/8]
                relative_coords_b = relative_coords[:, inst_st: inst_st + num_inst, :, :]
                mask_feats_b = mask_feats[i].reshape(1, in_channels, H * W).unsqueeze(1).repeat(1, num_inst, 1, 1)
                mask_head_b = torch.cat([relative_coords_b, mask_feats_b], dim=2)

                mask_head_inputs.append(mask_head_b)
                inst_st += num_inst

        else:
            mask_head_inputs = []
            inst_st = 0
            for i, num_inst in enumerate(num_insts):
                mask_head_b = mask_feats[i].reshape(1, in_channels, H * W).unsqueeze(1).repeat(1, num_inst, 1, 1)
                mask_head_b = mask_head_b.reshape(1, -1, H, W)
                mask_head_inputs.append(mask_head_b)

        # mask_head_inputs: [1, \sum{num_queries * (C/32+2)}, H/8, W/8]
        mask_head_inputs = torch.cat(mask_head_inputs, dim=1)
        mask_head_inputs = mask_head_inputs.reshape(1, -1, H, W)
        # mask_head_params: [num_insts_all, num_params]
        mask_head_params = torch.flatten(mask_head_params, 0, 1)

        if num_insts_all != 0:
            weights, biases = parse_dynamic_params(
                mask_head_params, self.dynamic_mask_channels,
                self.weight_nums, self.bias_nums
            )

            mask_logits = self.mask_heads_forward(mask_head_inputs, weights, biases, mask_head_params.shape[0])
        else:
            mask_logits = mask_head_inputs
            return mask_logits
        # mask_logits: [1, num_insts_all, H/8, W/8]
        mask_logits = mask_logits.reshape(-1, 1, H, W)
        # upsample predicted masks
        assert mask_feat_stride >= self.mask_out_stride
        assert mask_feat_stride % self.mask_out_stride == 0

        mask_logits = aligned_bilinear(mask_logits, int(mask_feat_stride / self.mask_out_stride))

        mask_logits = mask_logits.reshape(1, -1, mask_logits.shape[-2], mask_logits.shape[-1])
        # mask_logits: [1, num_insts_all, H/4, W/4]

        return mask_logits

    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_mask):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, 'pred_masks': c}
                for a, b, c  in zip(outputs_class[:,:-1,...].transpose(0,1),
                                    outputs_coord[:,:-1,...].transpose(0,1), outputs_mask[:,:-1,...].transpose(0,1))]

    def _set_enc_aux_loss(self, outputs_class, outputs_coord):
        outputs_class = [torch.cat([oc[i] for oc in outputs_class[1:]],1) for i in range(len(outputs_class[0]))]
        outputs_coord = [torch.cat([oc[i] for oc in outputs_coord[1:]],1) for i in range(len(outputs_coord[0]))]
        return [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class, outputs_coord)]


class MaskHeadSmallConv(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, dim, fpn_dims, context_dim):
        super().__init__()

        # inter_dims = [dim, context_dim // 2, context_dim // 4, context_dim // 8, context_dim // 16, context_dim // 64]
        inter_dims = [dim, context_dim, context_dim, context_dim, context_dim, context_dim]

        # used after upsampling to reduce dimention of fused features!
        self.lay1 = torch.nn.Conv2d(dim, dim // 4, 3, padding=1)
        self.lay2 = torch.nn.Conv2d(dim // 4, dim // 32, 3, padding=1)
        self.lay3 = torch.nn.Conv2d(inter_dims[1], inter_dims[2], 3, padding=1)
        self.lay4 = torch.nn.Conv2d(inter_dims[2], inter_dims[3], 3, padding=1)
        self.dcn = torch.nn.Conv2d(inter_dims[3], inter_dims[4], 3, padding=1)
        self.dim = dim

        if fpn_dims != None:
            self.adapter1 = torch.nn.Conv2d(fpn_dims[0], inter_dims[1], 1)
            self.adapter2 = torch.nn.Conv2d(fpn_dims[1], inter_dims[2], 1)
            self.adapter3 = torch.nn.Conv2d(fpn_dims[2], inter_dims[3], 1)

        for name, m in self.named_modules():
            if name == "conv_offset":
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
            else:
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_uniform_(m.weight, a=1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, fpns):
        if fpns != None:
            cur_fpn = self.adapter1(fpns[0])
            if cur_fpn.size(0) != x[-1].size(0):
                cur_fpn = _expand(cur_fpn, x[-1].size(0) // cur_fpn.size(0))
            fused_x = (cur_fpn + x[-1]) / 2
        else:
            fused_x = x[-1]
        fused_x = self.lay3(fused_x)
        fused_x = F.relu(fused_x)

        if fpns != None:
            cur_fpn = self.adapter2(fpns[1])
            if cur_fpn.size(0) != x[-2].size(0):
                cur_fpn = _expand(cur_fpn, x[-2].size(0) // cur_fpn.size(0))
            fused_x = (cur_fpn + x[-2]) / 2 + F.interpolate(fused_x, size=cur_fpn.shape[-2:], mode="nearest")
        else:
            fused_x = x[-2] + F.interpolate(fused_x, size=x[-2].shape[-2:], mode="nearest")
        fused_x = self.lay4(fused_x)
        fused_x = F.relu(fused_x)

        if fpns != None:
            cur_fpn = self.adapter3(fpns[2])
            if cur_fpn.size(0) != x[-3].size(0):
                cur_fpn = _expand(cur_fpn, x[-3].size(0) // cur_fpn.size(0))
            fused_x = (cur_fpn + x[-3]) / 2 + F.interpolate(fused_x, size=cur_fpn.shape[-2:], mode="nearest")
        else:
            fused_x = x[-3] + F.interpolate(fused_x, size=x[-3].shape[-2:], mode="nearest")
        fused_x = self.dcn(fused_x)
        fused_x = F.relu(fused_x)
        fused_x = self.lay1(fused_x)
        fused_x = F.relu(fused_x)
        fused_x = self.lay2(fused_x)
        fused_x = F.relu(fused_x)

        return fused_x


def _expand(tensor, length: int):
    return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)


def parse_dynamic_params(params, channels, weight_nums, bias_nums):
    assert params.dim() == 2
    assert len(weight_nums) == len(bias_nums)
    assert params.size(1) == sum(weight_nums) + sum(bias_nums)

    num_insts = params.size(0)
    num_layers = len(weight_nums)

    params_splits = list(torch.split_with_sizes(params, weight_nums + bias_nums, dim=1))

    weight_splits = params_splits[:num_layers]
    bias_splits = params_splits[num_layers:]

    for l in range(num_layers):
        if l < num_layers - 1:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
        else:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts)

    return weight_splits, bias_splits


def aligned_bilinear(tensor, factor):
    assert tensor.dim() == 4
    assert factor >= 1
    assert int(factor) == factor

    if factor == 1:
        return tensor

    h, w = tensor.size()[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(
        tensor, size=(oh, ow),
        mode='bilinear',
        align_corners=True
    )
    tensor = F.pad(
        tensor, pad=(factor // 2, 0, factor // 2, 0),
        mode="replicate"
    )

    return tensor[:, :, :oh - 1, :ow - 1]


def compute_locations(h, w, device, stride=1):
    shifts_x = torch.arange(
        0, w * stride, step=stride,
        dtype=torch.float32, device=device)

    shifts_y = torch.arange(
        0, h * stride, step=stride,
        dtype=torch.float32, device=device)

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    return locations


def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


class PostProcessSegm(nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, results, outputs, orig_target_sizes, max_target_sizes):
        # output single / multi frames
        assert len(orig_target_sizes) == len(max_target_sizes)
        # max_h, max_w = max_target_sizes.max(0)[0].tolist()

        # pred_logits: [bs, num_querries, num_classes]
        # pred_masks: [bs, num_querries, num_frames, H/8, W/8]

        out_refs = outputs['reference_points']
        outputs_masks = outputs["pred_masks"]
        out_logits = outputs['pred_logits']

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]

        outputs_masks = [out_m[topk_boxes[i]].unsqueeze(0) for i, out_m in enumerate(outputs_masks)]
        outputs_masks = torch.cat(outputs_masks)
        bs, _, num_frames, H, W = outputs_masks.shape

        # outputs_masks = F.interpolate(outputs_masks.flatten(0,1), size=(max_h, max_w), mode="bilinear", align_corners=False)
        outputs_masks = F.interpolate(outputs_masks.flatten(0, 1), size=(H * 4, W * 4), mode="bilinear",
                                      align_corners=False)
        # outputs_masks = (outputs_masks.sigmoid() > self.threshold).cpu()
        outputs_masks = outputs_masks.sigmoid() > self.threshold

        # [bs, num_frames, 10, H, W]
        outputs_masks = outputs_masks.reshape(bs, -1, num_frames, outputs_masks.shape[-2],
                                              outputs_masks.shape[-1]).permute(0, 2, 1, 3, 4)

        # reference points for each instance
        references = [refs[topk_boxes[i]].unsqueeze(0) for i, refs in enumerate(out_refs)]
        references = torch.cat(references)

        for i, (cur_mask, t, tt) in enumerate(zip(outputs_masks, max_target_sizes, orig_target_sizes)):
            img_h, img_w = t[0], t[1]
            results[i]["scores"] = scores[i]
            results[i]["labels"] = labels[i]
            results[i]['reference_points'] = references[i]

            results[i]["masks"] = cur_mask[:, :, :img_h, :img_w]
            results[i]["masks"] = F.interpolate(results[i]["masks"].float(), size=tuple(tt.tolist()),
                                                mode="nearest").byte()
            results[i]["masks"] = results[i]["masks"].permute(1, 0, 2, 3)

        # required dim of results:
        #   scores: [num_ins]
        #   labels: [num_ins]
        #   reference_points: [num_ins, num_frames, 2]
        #   masks: [num_ins, num_frames, H, W]

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
