import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import util.box_ops as box_ops
from util.misc import NestedTensor, get_top_mem_idx, nested_tensor_from_tensor_list, inverse_sigmoid

def run_backbone(self, samples, train=True):
    ''' Run backbone features with frame and other reshaping'''

    features, pos = self.detr.backbone(samples)
    srcs = []
    masks = []
    poses = []
    spatial_shapes = []
    for l, feat in enumerate(features[1:]):
        # src: [nf*N, _C, Hi, Wi],
        # mask: [nf*N, Hi, Wi],
        # pos: [nf*N, C, H_p, W_p]
        src, mask = feat.decompose()
        src_proj_l = self.detr.input_proj[l](src)  # src_proj_l: [nf*N, C, Hi, Wi]
        pos_l = pos[l + 1]
        # src_proj_l -> [nf*N, C, Hi, Wi]
        n, c, h, w = src_proj_l.shape
        spatial_shapes.append((h, w))
        if train:
            src_proj_l = src_proj_l.reshape(n // self.num_frames, self.num_frames, c, h, w)

            # mask -> [nf*N, Hi, Wi]
            mask = mask.reshape(n // self.num_frames, self.num_frames, h, w)

            # pos -> [nf*N, Hi, Wi]
            np, cp, hp, wp = pos_l.shape
            pos_l = pos_l.reshape(np // self.num_frames, self.num_frames, cp, hp, wp)

        srcs.append(src_proj_l)
        masks.append(mask)
        poses.append(pos_l)
        assert mask is not None

    if self.detr.num_feature_levels > (len(features) - 1):
        _len_srcs = len(features) - 1
        for l in range(_len_srcs, self.detr.num_feature_levels):
            if l == _len_srcs:
                src = self.detr.input_proj[l](features[-1].tensors)
            else:
                src = self.detr.input_proj[l](srcs[-1])
            m = samples.mask  # [nf*N, H, W]
            mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
            pos_l = self.detr.backbone[1](NestedTensor(src, mask)).to(src.dtype)

            # src -> [nf*N, C, H, W]
            n, c, h, w = src.shape
            spatial_shapes.append((h, w))
            np, cp, hp, wp = pos_l.shape
            if train:
                src = src.reshape(n // self.num_frames, self.num_frames, c, h, w)
                mask = mask.reshape(n // self.num_frames, self.num_frames, h, w)
                pos_l = pos_l.reshape(np // self.num_frames, self.num_frames, cp, hp, wp)
            srcs.append(src)
            masks.append(mask)
            poses.append(pos_l)

    return srcs, masks, poses, spatial_shapes, np, c


def transformer_postprocess(self, hs, memory, init_ref, inter_ref, s_shapes, frame, orig_w=None, orig_h=None,
                            top_mem=None, final_only=False, t_classes=None, valid_ratios=None,criterion=None, indices_list=None,gt_targets=None ):
    o_classes = []
    o_cord = []
    o_maks = []
    enc_lay_num = hs.shape[0]
    for lvl in range(enc_lay_num):
        if final_only:
            lvl = 5
        if lvl == 0:
            reference = init_ref[..., :self.detr.num_queries, :]
        else:
            reference = inter_ref[lvl - 1][..., :self.detr.num_queries, :]
        reference = inverse_sigmoid(reference)

        # if self.propagate_class and frame!=0:
        #     # temporal_class = self.detr.temporal_class_embed[lvl](t_classes[lvl])
        #     # outputs_class = self.detr.class_embed[lvl](hs[lvl])
        #     # outputs_class *= temporal_class.softmax(2) #softmax per query
        #     time_weight = self.detr.class_embed[lvl](hs[lvl])
        #     outputs_class = t_classes[lvl] * time_weight
        # else:
        outputs_class = self.detr.class_embed[lvl](hs[lvl][..., :self.detr.num_queries, :])
        if self.propagate_class:
            inter_class = outputs_class.unsqueeze(1)
            if frame > 0:
                inter_class = torch.cat((torch.stack(t_classes).permute(1, 0, 2, 3), inter_class), dim=1)  # hardcode to past 5, change to mem token
            temporal_class = self.detr.temporal_class_embed[lvl](inter_class).sigmoid()
            outputs_class = (outputs_class.unsqueeze(1) * F.softmax(temporal_class, 1)).sum(1)

        tmp = self.detr.bbox_embed[lvl](hs[lvl][..., :self.detr.num_queries, :])  # here also same embedding

        if reference.shape[-1] == 4:
            tmp += reference.squeeze(1)
        else:
            assert reference.shape[-1] == 2
            tmp[..., :2] += reference.squeeze(1)
        outputs_coord = tmp.sigmoid()
        o_classes.append(outputs_class)
        o_cord.append(outputs_coord)
        outputs_layer = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}
        dynamic_mask_head_params = self.controller(hs[lvl][..., :self.detr.num_queries, :])  # [bs, num_quries, num_params]

        if not final_only: # only for training
            if frame == 0 : #or lvl!=enc_lay_num-1:
                # for training & log evaluation loss
                indices = criterion.matcher(outputs_layer, gt_targets, frame, valid_ratios, online=True)
                if frame==0:
                    indices_list.append(indices)
            else:
                indices = indices_list[-1]  # intially indicies sud reamin same

            reference_points, mask_head_params, num_insts = [], [], []
            for i, indice in enumerate(indices):
                pred_i, tgt_j = indice
                num_insts.append(len(pred_i))
                mask_head_params.append(dynamic_mask_head_params[i, pred_i].unsqueeze(0))

                # This is the image size after data augmentation (so as the gt boxes & masks)

                orig_h, orig_w = gt_targets[i]['size']
                scale_f = torch.stack([orig_w, orig_h], dim=0)

                ref_cur_f = reference[i].sigmoid()
                ref_cur_f = ref_cur_f[..., :2]
                ref_cur_f = ref_cur_f * scale_f[None, None, :]

                reference_points.append(ref_cur_f[:, pred_i].unsqueeze(0))
            # reference_points: [1, nf,  \sum{selected_insts}, 2]
            # mask_head_params: [1, \sum{selected_insts}, num_params]
            reference_points = torch.cat(reference_points, dim=2)
            dynamic_mask_head_params = torch.cat(mask_head_params, dim=1)


        if final_only: #only for validation
            orig_w = torch.tensor(orig_w).to(reference)
            orig_h = torch.tensor(orig_h).to(reference)
            scale_f = torch.stack([orig_w, orig_h], dim=0)
            reference_points = reference[..., :2].sigmoid() * scale_f[None, None, None, :]
            num_insts = [self.detr.num_queries]

        # mask prediction
        outputs_layer = self.forward_mask_head_train(outputs_layer, memory, s_shapes,
                                                     reference_points, dynamic_mask_head_params, num_insts)
        o_maks.append(torch.cat(outputs_layer['pred_masks'], dim=1))

        # if self.reinitialize_query:
        #     topkv, top_indices = torch.topk(outputs_class[0, ...].sigmoid().cpu().detach().flatten(0), k=10)
        #     top_indices = top_indices.tolist()
        #     hit_dict = [idx // 42 for idx in top_indices]
        #     all_idx = list(range(300))
        #     hit_dict = [item for item in all_idx if item not in hit_dict]
        #     hit_dict = torch.tensor(hit_dict).to(tgt.device)
        #     hs[-1, :, hit_dict, :] = tgt[:, hit_dict, :]

        if final_only:  # take only the final layer
            break

    if self.propagate_class:
        t_classes.append(outputs_class.detach() if self.sg else outputs_class)  # only for final layer
        if len(t_classes) > self.detr.mframe and self.dynamic_memory:
            del t_classes[0]

    supcon_indices = None
    if self.dynamic_memory:
        if final_only:
            top_mem, supcon_indices = get_top_mem_idx(o_classes[-1], self.detr.mtoken, memory=hs[-1, ...], pred_boxes=o_cord[-1], nf=frame, val=True)
        else: #for training
            top_mem, supcon_indices = get_top_mem_idx(o_classes[-1], self.detr.mtoken, indices_list[-1], memory=hs[-1, ...],targets=gt_targets,
                                       pred_boxes=o_cord[-1], nf=frame,gt_usage=self.gt_usage)

        # hs_mem.append(top_mem)
        # if len(hs_mem) > self.detr.mframe - 1:  # delte old memeories
        #     del hs_mem[0]

    return torch.stack(o_classes), torch.stack(o_cord), torch.cat(o_maks, dim=0), top_mem, t_classes, supcon_indices

def get_memories(self,frame,hs_mem,memory_pos, hs=None, mem_query=None,hs_mem_pos=None,
                 token_memory=None,mem_mask=None, frame_mem=None, frame_mem_mask=None):
    frame_mem_pos = None
    if self.dynamic_memory:
        if frame == 0:
            if self.memory_as_query:
                frame_mem = mem_query
        else:
            prev_mem = self.detr.transformer.mem_token_proj(torch.stack(hs_mem))
            ##nf,bs,mtoken,c->bs,c,nf*mtoken
            if self.memory_as_query:
                frame_mem = torch.cat((prev_mem, hs[-1][..., self.detr.num_queries:, :].unsqueeze(0)), dim=0).permute(1,0,2,3).flatten(1, 2)
                frame_mem_pos = memory_pos[..., :frame + 1, :].flatten(-2).permute(0, 2,1)
            else:
                frame_mem = prev_mem.permute(1, 0, 2, 3).flatten(1, 2)
                frame_mem_pos = memory_pos[..., :frame, :].flatten(-2).permute(0, 2, 1)
                if self.temp_midx_pos: # add sin emb
                     frame_mem_pos = torch.stack(hs_mem_pos).permute(1, 0, 2, 3).flatten(1, 2)+ frame_mem_pos.repeat_interleave(self.detr.mtoken,dim=1)
    else:
        frame_mem = token_memory[frame, ...]
        frame_mem_pos = memory_pos[frame, ...]
        frame_mem_mask = mem_mask[frame, ...]

    return frame_mem, frame_mem_pos, frame_mem_mask



def run_transformer(self, frame, src, mask, src_pos, tgt, query_pos, spatial_shapes, criterion=None, gt_targets = None, init_reference =None, memory =None, memory_pos=None,
                    memory_mask=None, mem_query=None, mem_query_pos=None, hs=None, inter_references =None, save_memory= False):
    t_classes = []
    indices_list = []
    outputs_classes = []
    outputs_coords = []
    outputs_mask = []
    hss = []
    idxs = []

    hs, hs_box, memory, init_reference, inter_references, inter_samples, enc_outputs_class, valid_ratios = \
        self.detr.transformer(src, mask, src_pos, query_pos,
                              tgt if frame == None else hs[-1, ...].detach() if self.sg else hs[-1, ...],
                              (init_reference.detach() if self.sg else init_reference) if
                              (self.propagate_ref_points and init_reference is not None) else None,
                              (inter_references.detach() if self.sg else inter_references) if
                              (self.propagate_inter_ref and inter_references is not None) else None,
                              memory, memory_pos, memory_mask, mem_query, mem_query_pos)
    hs = hs.squeeze(2)

    valid_ratios = valid_ratios[:, 0]  #
    # memory: [bz,n_f, \sigma(HiWi), C]
    # hs: [num_encoders, N, num_querries, C]
    o_classes = []
    o_cord = []
    o_maks = []
    enc_lay_num = hs.shape[0]
    for lvl in range(enc_lay_num):
        if lvl == 0:
            reference = init_reference
        else:
            reference = inter_references[lvl - 1]
        reference = inverse_sigmoid(reference)

        # if self.propagate_class and frame!=0:
        #     temporal_class = self.detr.temporal_class_embed[lvl](t_classes[lvl])
        #     outputs_class = self.detr.class_embed[lvl](hs[lvl])
        #     outputs_class += temporal_class.sigmoid() #softmax per query
        #     # time_weight = self.detr.class_embed[lvl](hs[lvl])
        #     # outputs_class = t_classes[lvl] * time_weight
        # else:
        outputs_class = self.detr.class_embed[lvl](hs[lvl])
        if self.propagate_class:
            inter_class = outputs_class.unsqueeze(1)
            if frame > 0:
                inter_class = torch.cat((torch.stack(t_classes).permute(1, 0, 2, 3), inter_class), dim=1)
            temporal_class = self.detr.temporal_class_embed[lvl](inter_class)
            outputs_class = (outputs_class.unsqueeze(1) * F.softmax(temporal_class, 1)).sum(1)

        tmp = self.detr.bbox_embed[lvl](hs[lvl])  # here also same embedding

        if reference.shape[-1] == 4:
            tmp += reference.squeeze(1)
        else:
            assert reference.shape[-1] == 2
            tmp[..., :2] += reference.squeeze(1)
        outputs_coord = tmp.sigmoid()
        o_classes.append(outputs_class)
        o_cord.append(outputs_coord)
        outputs_layer = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}

        dynamic_mask_head_params = self.controller(hs[lvl])  # [bs, num_quries, num_params]

        if frame == 0 or save_memory:
            # for training & log evaluation loss
            indices = criterion.matcher(outputs_layer, gt_targets, frame, valid_ratios, online=True)
            indices_list.append(indices)
        else:
            indices = indices_list[lvl]  # intially indicies sud reamin same

        reference_points, mask_head_params, num_insts = [], [], []
        for i, indice in enumerate(indices):
            pred_i, tgt_j = indice
            num_insts.append(len(pred_i))
            mask_head_params.append(dynamic_mask_head_params[i, pred_i].unsqueeze(0))

            # This is the image size after data augmentation (so as the gt boxes & masks)

            orig_h, orig_w = gt_targets[i]['size']
            scale_f = torch.stack([orig_w, orig_h], dim=0)

            ref_cur_f = reference[i].sigmoid()
            ref_cur_f = ref_cur_f[..., :2]
            ref_cur_f = ref_cur_f * scale_f[None, None, :]

            reference_points.append(ref_cur_f[:, pred_i].unsqueeze(0))

        # reference_points: [1, nf,  \sum{selected_insts}, 2]
        # mask_head_params: [1, \sum{selected_insts}, num_params]
        reference_points = torch.cat(reference_points, dim=2)
        mask_head_params = torch.cat(mask_head_params, dim=1)

        # mask prediction
        outputs_layer = self.forward_mask_head_train(outputs_layer, memory, spatial_shapes,
                                                     reference_points, mask_head_params, num_insts)
        o_maks.append(torch.cat(outputs_layer['pred_masks'], dim=1))

    if self.propagate_class:
        t_classes.append(o_classes[-1].detach() if self.sg else o_classes[-1])
    outputs_classes.append(torch.stack(o_classes))
    outputs_coords.append(torch.stack(o_cord))
    outputs_mask.append(torch.cat(o_maks, dim=0))

    if save_memory:
        hss.append(hs[-1, ...])
        idxs.append(indices_list[-1])
