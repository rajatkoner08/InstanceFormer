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


import copy
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from util.misc import inverse_sigmoid
from models.ops.modules import MSDeformAttn
from util.dam import attn_map_to_flat_grid


class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_frames=1, num_feature_levels=4,
                 dec_n_points=4, enc_n_points=4, online=True, memory_frames=False, num_classes=42,
                 propagate_ref_points=False, propagate_time_ref=False, last_ref=False, alternate_attn=False,
                 attn_at=False, init_ref_after_memory=False, enc_ref_prop=False, num_queries=300,mask_query=False,
                 memory_as_query=False, dec_ref_prop=False,dec_ref_keep_ratio=0.5, enc_aux_loss=False, train=True,
                 analysis=False, propagate_inter_ref_2=False, propagate_ref_additive=False):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.num_feature_levels = num_feature_levels
        self.online = online
        self.propagate_ref_points = propagate_ref_points
        self.memory_frames = memory_frames
        self.propagate_time_ref = propagate_time_ref
        self.last_ref = last_ref
        self.alternate_attn = alternate_attn
        self.attn_at = attn_at
        self.init_ref_after_memory = init_ref_after_memory
        self.enc_ref_prop = enc_ref_prop
        self.memory_as_query = memory_as_query
        self.num_queries=num_queries
        self.dec_ref_prop=dec_ref_prop
        self.enc_aux_loss = enc_aux_loss
        self.dec_ref_keep_ratio=dec_ref_keep_ratio
        self.training = train
        self.analysis=analysis
        self.propagate_inter_ref_2=propagate_inter_ref_2
        self.propagate_ref_additive=propagate_ref_additive
        if self.memory_frames >0:
            self.mem_token_proj = nn.Linear(d_model+4+num_classes, d_model) # a linear projection of memory token
            self.mem_token_proj_drp = nn.Dropout(dropout)

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels , 
                                                          nhead, enc_n_points)

        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers, enc_ref = self.enc_ref_prop,
                                                        dec_ref_keep_ratio=dec_ref_keep_ratio)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels,
                                                          nhead, dec_n_points, self.online, memory=memory_frames,
                                                          alternate_attn=self.alternate_attn, attn_at=self.attn_at,
                                                          init_ref_after_memory=self.init_ref_after_memory,
                                                          num_queries=self.num_queries, mask_query=mask_query,
                                                          memory_as_query=self.memory_as_query)

        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, num_frames, self.online,
                                return_intermediate_dec, self.init_ref_after_memory, self.dec_ref_prop, self.analysis)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self.reference_points = nn.Linear(d_model, 2)
        if self.propagate_time_ref:
            self.time_references_weights = nn.Linear(d_model, 2)
        self.num_feature_levels = num_feature_levels
        self.dec_n_points = dec_n_points
        self.num_queries = num_queries
        self.num_decoder_layers=num_decoder_layers
        if self.dec_ref_prop:
            self.dec_to_enc_ref = nn.Linear(d_model, self.nhead*self.num_feature_levels*num_decoder_layers*dec_n_points*2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        
        xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        """Make region proposals for each multi-scale features considering their shapes and padding masks,
        and project & normalize the encoder outputs corresponding to these proposals.
            - center points: relative grid coordinates in the range of [0.01, 0.99] (additional mask)
            - width/height:  2^(layer_id) * s (s=0.05) / see the appendix A.4

        Tensor shape example:
            Args:
                memory: torch.Size([2, 15060, 256])
                memory_padding_mask: torch.Size([2, 15060])
                spatial_shape: torch.Size([4, 2])
            Returns:
                output_proposals: torch.Size([2, 15060, 4])
                    - x, y, w, h
        """
        N_, S_, C_ = memory.shape
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            # level of encoded feature scale
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)

        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))  # inverse of sigmoid
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))  # sigmoid(inf) = 1

        return output_proposals

    def forward(self, srcs, masks, pos_embeds, query_embed=None, tgt=None, init_reference=None, inter_references=None, mem_token =None,
                mem_token_pos=None, mem_token_mask=None, mem_query =None, mem_query_pos=None, enc_sample_loc =None, dec_sample_loc=None):
        assert  query_embed is not None
        # srcs: 4(N, C, Hi, Wi)
        # query_embed: [300, C] 
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, nf, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = src.flatten(3).transpose(2, 3)   # src: [N, nf, Hi*Wi, C]
            mask = mask.flatten(2)   # mask: [N, nf, Hi*Wi]
            pos_embed = pos_embed.flatten(3).transpose(2, 3)  # pos_embed: [N, nf, Hp*Wp, C]
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, 1, -1)

            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        # src_flatten: [\sigma(N*Hi*Wi), C]
        src_flatten = torch.cat(src_flatten, 2)
        mask_flatten = torch.cat(mask_flatten, 2)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 2)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m[:,0]) for m in masks], 1)  

        # encoder
        if dec_sample_loc is not None:
            dec_sample_loc[0] = self.dec_ref_offset(dec_sample_loc[0], query_embed)
        if self.enc_aux_loss and self.training:
            output_proposals = self.gen_encoder_output_proposals((src_flatten + lvl_pos_embed_flatten)[:,0,...],
                                                mask_flatten[:,0,...], spatial_shapes) if self.enc_aux_loss else None
        else:
            output_proposals = None

        memory, enc_sample_loc, topk, enc_inter_outputs_class, enc_inter_outputs_coord_unact\
                                        = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios,
                                              lvl_pos_embed_flatten, mask_flatten, enc_sample_loc, dec_sample_loc, output_proposals)
        # src_flatten,lvl_pos_embed_flatten shape= [bz, nf, 4lvl*wi*hi, C]    mask_flatten: [bz, nf, 4lvl*wi*hi]  


        # prepare input for decoder
        bs, nf,  _, c = memory.shape

        if self.propagate_inter_ref_2:
            reference_points = self.init_referance_2(init_reference, nf, query_embed)
        else:
            reference_points = self.init_referance(init_reference, nf, query_embed)

        if not isinstance(reference_points, tuple):
            reference_points = reference_points.unsqueeze(1).repeat(1,nf,1,1)     #[bz,nf,300,2]
            init_reference_out = reference_points

        # decoder
        hs, hs_box, inter_references, inter_samples, reference_out, mem_query, mem_query_pos, dec_sample_loc_new, sampling_locations_all= self.decoder(tgt, reference_points, memory, spatial_shapes, level_start_index,
                                                                   valid_ratios, query_embed, mask_flatten, inter_references, mem_token,
                                                                   mem_token_pos, mem_token_mask, mem_query, mem_query_pos,)
        if self.init_ref_after_memory and reference_out is not None:
            init_reference_out=reference_out

        return hs, hs_box, memory, init_reference_out, inter_references, inter_samples, None, valid_ratios,\
               enc_sample_loc, mem_query, mem_query_pos, dec_sample_loc, dec_sample_loc_new, level_start_index, topk, \
               enc_inter_outputs_class, enc_inter_outputs_coord_unact, sampling_locations_all

    def dec_ref_offset(self, dec_sample_loc, query_embed):
        if self.dec_ref_prop and dec_sample_loc is not None:
            offset = self.dec_to_enc_ref(query_embed)
            offset = offset.view([-1,self.num_decoder_layers,self.num_queries,self.nhead,self.num_feature_levels,self.dec_n_points, 2])
            dec_sample_loc = dec_sample_loc.clone().detach() + offset
        return dec_sample_loc

    def init_referance(self, init_reference, nf, query_embed):
        if init_reference is not None:
            if self.init_ref_after_memory:
                if self.propagate_time_ref:
                    reference_points = (self.time_references_weights(init_reference[:, 0, ...]), self.reference_points, nf)
                else:
                    reference_points = (init_reference[:, 0, ...], self.reference_points, nf)
            else:
                time_weight = self.reference_points(query_embed).sigmoid()
                if self.propagate_time_ref:
                    time_ref = self.time_references_weights(init_reference[:, 0, ...])
                    reference_points = (time_ref * time_weight).sigmoid()
                else:
                    if self.propagate_ref_additive:
                        reference_points = (init_reference[:, 0, ...] + time_weight).sigmoid()
                    else:
                        reference_points = (init_reference[:, 0, ...] * time_weight).sigmoid()
        else:
            reference_points = self.reference_points(query_embed).sigmoid()
        return reference_points

    def init_referance_2(self, init_reference, nf, query_embed):
        offset = self.time_references_weights(query_embed)
        if init_reference is not None:
            reference_points = (inverse_sigmoid(init_reference[:, 0, ...]) + offset).sigmoid()
        else: # frame 0
            reference_points = (self.reference_points(query_embed) + offset*0).sigmoid()
        return reference_points


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points, 'encode')
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None, pre_sample_ofst=False, tgt=None):
        # self attention
        if tgt is None:
            src2, sampling_locations = self.self_attn(self.with_pos_embed(src, pos), None, reference_points, src, spatial_shapes,
                                                      level_start_index, padding_mask, pre_sample_ofst = pre_sample_ofst)
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            # ffn
            src = self.forward_ffn(src)
            return src, sampling_locations
        else:
            # self attention
            tgt2, sampling_locations = self.self_attn(self.with_pos_embed(tgt, pos), None,
                                reference_points, src, spatial_shapes,
                                level_start_index, padding_mask, pre_sample_ofst = pre_sample_ofst)
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

            # ffn
            tgt = self.forward_ffn(tgt)

            return tgt, sampling_locations


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, enc_ref,dec_ref_keep_ratio=0.5):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.enc_ref = enc_ref
        self.dec_ref_keep_ratio = dec_ref_keep_ratio
        self.enc_aux_loss = False
        self.bbox_embed = None
        self.class_embed = None
        #self.pre_sample_ofst = None

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def sparse_query_index(self, dec_sample_loc, level_start_index, spatial_shapes, src):
        flat_grid = attn_map_to_flat_grid(spatial_shapes, level_start_index, dec_sample_loc[0], dec_sample_loc[1]).sum(dim=(1,2))
        topk = int(src.shape[2] * self.dec_ref_keep_ratio)
        topk_idx = torch.topk(flat_grid, topk)[1]
        return topk_idx, topk

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None,
                pre_sample_loc=None, dec_sample_loc=None, output_proposals=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        # print(dec_sample_loc is not None)
        # start1 = time.process_time()
        topk_inds=None
        topk = None

        enc_inter_outputs_class = []
        enc_inter_outputs_coords = []

        if dec_sample_loc is not None:
            topk_inds, topk = self.sparse_query_index(dec_sample_loc, level_start_index, spatial_shapes, src)
            B_, N_, S_, P_ = reference_points.shape
            reference_points = torch.gather(reference_points.view(B_, N_, -1), 1, topk_inds.unsqueeze(-1).repeat(1, 1, S_*P_)).view(B_, -1, S_, P_)
            tgt = torch.gather(output[:,0,...], 1, topk_inds.unsqueeze(-1).repeat(1, 1, output.size(-1)))[:,None,...]
            pos = torch.gather(pos[:,0,...], 1, topk_inds.unsqueeze(-1).repeat(1, 1, pos.size(-1)))[:,None,...]
            if output_proposals is not None:
                output_proposals = output_proposals.gather(1, topk_inds.unsqueeze(-1).repeat(1, 1, output_proposals.size(-1)))
        else:
            tgt = None
        for lid, layer in enumerate(self.layers):
            tgt, sampling_locations = layer(output, pos, reference_points, spatial_shapes, level_start_index,
                                            padding_mask, pre_sample_ofst=pre_sample_loc, tgt=tgt)
            if dec_sample_loc is not None:
                outputs = []
                for i in range(topk_inds.shape[0]):
                    outputs.append(output[i,0].scatter(0, topk_inds[i].unsqueeze(-1).repeat(1, tgt.size(-1)), tgt[i,0]))
                output = torch.stack(outputs)[:,None,...]
            else:
                output = tgt

            if self.enc_aux_loss and lid < self.num_layers - 1 and output_proposals is not None:
                # feed outputs to aux. heads
                output_class = self.class_embed[lid](tgt)
                output_offset = self.bbox_embed[lid](tgt)
                output_coords_unact = output_proposals[:,None,...] + output_offset
                # values to be used for loss compuation
                enc_inter_outputs_class.append(output_class)
                enc_inter_outputs_coords.append(output_coords_unact.sigmoid())

        # print(time.process_time() - start1)
        return output, sampling_locations if self.enc_ref else None, topk, enc_inter_outputs_class, enc_inter_outputs_coords


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, online=False, memory=False, alternate_attn=False, attn_at=False,
                 init_ref_after_memory=False,num_queries=300, mask_query =False, memory_as_query=False):
        super().__init__()
        self.online = online # for online or seq attn
        self.num_queries = num_queries
        self.mask_query = mask_query
        self.query_mask = torch.zeros(num_queries,dtype=bool,requires_grad=False)
        self.num_mask = int(num_queries*dropout/3)
        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points, 'decode', online=self.online)
        self.dropout1 = nn.Dropout(dropout)
        # self.norm1 = nn.LayerNorm(d_model)
        self.dropout1_box = nn.Dropout(dropout)
        self.norm1_box = nn.LayerNorm(d_model)

        # self attention for mask&class query
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.memory_as_query = memory_as_query
        if memory>0 or self.memory_as_query:
            self.cross_attn_mem = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
            self.dropout2_mem = nn.Dropout(dropout)
            self.norm2_mem = nn.LayerNorm(d_model)
        if self.memory_as_query:
            self.cross_attn_mem_inverse = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
            self.dropout2_mem_inverse = nn.Dropout(dropout)
            self.norm2_mem_inverse = nn.LayerNorm(d_model)

        # ffn for memory
        self.alter = alternate_attn
        self.attn_at = attn_at
        self.init_ref_after_memory=init_ref_after_memory
        if alternate_attn:
            self.linear1 = nn.Linear(d_model, d_ffn)
            self.activation = _get_activation_fn(activation)
            self.dropout3 = nn.Dropout(dropout)
            self.linear2 = nn.Linear(d_ffn, d_model)
            self.dropout4 = nn.Dropout(dropout)
            self.norm3 = nn.LayerNorm(d_model)


        # ffn for box
        self.linear1_box = nn.Linear(d_model, d_ffn)
        self.activation_box = _get_activation_fn(activation)
        self.dropout3_box = nn.Dropout(dropout)
        self.linear2_box = nn.Linear(d_ffn, d_model)
        self.dropout4_box = nn.Dropout(dropout)
        self.norm3_box = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    @staticmethod
    def with_pos_embed_multf(tensor, pos):  # boardcase pos to every frame features
        return tensor if pos is None else tensor + pos.unsqueeze(1)

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_ffn_box(self, tgt):
        tgt2 = self.linear2_box(self.dropout3_box(self.activation_box(self.linear1_box(tgt))))
        tgt = tgt + self.dropout4_box(tgt2)
        tgt = self.norm3_box(tgt)
        return tgt

    def forward(self, tgt, tgt_box, query_pos, reference_points, src, src_spatial_shapes, level_start_index,
                src_padding_mask=None, mem = None, mem_pos = None, mem_token_mask=None,  mem_query=None, mem_query_pos=None, lid=0):
       if self.memory_as_query and lid < 3:
            tgt = self.memory_attn(mem_query, mem_query_pos, mem_token_mask, query_pos, tgt)

       if (mem != None and self.attn_at==1 and not self.alter) or (self.alter and mem != None and lid%2!=0 and self.attn_at==1):
            tgt = self.memory_attn(mem, mem_pos, mem_token_mask, query_pos, tgt)

        # self attention
       elif not (self.alter and mem != None and lid%2!=0):
            q1 = k1 = self.with_pos_embed(tgt, query_pos)
            tgt2 = self.self_attn(q1.transpose(0, 1), k1.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)


       if (mem != None and self.attn_at==2 and not self.alter) or (self.alter and mem != None and lid%2!=0 and self.attn_at==2):
           tgt = self.memory_attn(mem, mem_pos, mem_token_mask, query_pos, tgt)

       if self.init_ref_after_memory and isinstance(reference_points, tuple):
           init_reference_out, reference_points = self.init_ref_after_mem_attn(query_pos, reference_points)

       else:
           init_reference_out=None

       # cross attention
       tgt2, _ , sampling_locations, attention_weights = self.cross_attn(tgt[:,:self.num_queries,:], self.with_pos_embed(tgt[:,:self.num_queries,:], query_pos[:,:self.num_queries,:]),
                                                                         reference_points[:,:,:self.num_queries,...], src, src_spatial_shapes,
                                                                         level_start_index, src_padding_mask)
       tgt2 = tgt2.squeeze(1)
       tgt[:,:self.num_queries,:] = tgt[:,:self.num_queries,:] + self.dropout1_box(tgt2)
       tgt = self.norm1_box(tgt)



       if (mem != None and self.attn_at==3 and not self.alter) or (self.alter and mem != None and lid%2!=0  and self.attn_at==3):
           tgt = self.memory_attn(mem, mem_pos, mem_token_mask, query_pos, tgt)


       if self.alter and mem != None and lid%2!=0 :
           tgt = self.forward_ffn(tgt)
       else:  # ffn box
           tgt = self.forward_ffn_box(tgt)

       if self.memory_as_query and lid>=3:
           mem_query = self.memory_attn_inverse(tgt, query_pos, mem_token_mask, mem_query_pos, mem_query)

       # inflate dimesion to frame
       tgt = tgt.unsqueeze(1)

       return tgt, None, sampling_locations, attention_weights, init_reference_out, mem_query, mem_query_pos

    def memory_attn(self, mem, mem_pos, mem_token_mask, query_pos, tgt):
        tgt2 = self.cross_attn_mem(query=self.with_pos_embed(tgt, query_pos).permute(1, 0, 2),
                                   key=self.with_pos_embed(mem, mem_pos),
                                   value=mem, attn_mask=None, key_padding_mask=mem_token_mask)[0]
        tgt = tgt + self.dropout2_mem(tgt2.permute(1, 0, 2))
        tgt = self.norm2_mem(tgt)
        return tgt

    def memory_attn_inverse(self, mem, mem_pos, mem_token_mask, query_pos, tgt):
        tgt2 = self.cross_attn_mem_inverse(query=self.with_pos_embed(tgt, query_pos).permute(1, 0, 2),
                                   key=self.with_pos_embed(mem, mem_pos),
                                   value=mem, attn_mask=None, key_padding_mask=mem_token_mask)[0]
        tgt = tgt + self.dropout2_mem_inverse(tgt2.permute(1, 0, 2))
        tgt = self.norm2_mem_inverse(tgt)
        return tgt

    def init_ref_after_mem_attn(self, query_pos, reference_points):
        (reference_points, function, nf), src_valid_ratios = reference_points
        time_weight = function(query_pos).sigmoid()
        reference_points = (reference_points * time_weight).sigmoid()
        if reference_points.ndim == 3:
            reference_points = reference_points.unsqueeze(1).repeat(1, nf, 1, 1)  # [bz,nf,300,2]
        init_reference_out = reference_points
        if reference_points.shape[-1] == 4:
            reference_points = reference_points[:, :, :, None] \
                               * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None, None]
        else:
            assert reference_points.shape[-1] == 2
            reference_points = reference_points[:, :, :, None] * src_valid_ratios[:, None, None]
            # reference_points_input [bz, nf, 300, 4, 2]
        return init_reference_out, reference_points


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, num_frames, online=False, return_intermediate=False, init_ref_after_memory=False, dec_ref_prop=False, analysis=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.bbox_embed = None
        self.class_embed = None
        self.num_frames = num_frames
        self.online = online  #for online seq attn
        self.init_ref_after_memory = init_ref_after_memory
        self.dec_ref_prop=dec_ref_prop
        self.analysis = analysis

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,query_pos=None, src_padding_mask=None,
                 inter_references=None,  mem = None, mem_pos = None, mem_token_mask =None,mem_query=None, mem_query_pos=None):
        output = tgt
        intermediate = [] # save mask&class query in each decoder layers
        intermediate_box = []  # save box query 
        intermediate_reference_points = []
        intermediate_samples = []

        # reference_points： [bz, nf, 300, 2]
        # src: [2, nf, len_q, 256] encoder output
        #if not self.online:
        output_box = tgt   # box and mask&class share the same initial tgt, but perform deformable attention across frames independently,
        # before first decoder layer, output_box is  [bz,300,C]
        # after the first deformable attention, output_box becomes [bz, nf, 300, C] and keep shape between each decoder layers    
        attn_weights_all = []
        sampling_locations_all = []
        #for frame in range(self.num_frames):
        init_reference_out=None
        for lid, layer in enumerate(self.layers):
            if lid==0 and self.init_ref_after_memory and isinstance(reference_points, tuple):
                reference_points_input = (reference_points,src_valid_ratios)
            else:
                if reference_points.shape[-1] == 4:
                    reference_points_input = reference_points[:, :, :, None] * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None, None]
                else:
                    assert reference_points.shape[-1] == 2
                    reference_points_input = reference_points[:, :, :, None] * src_valid_ratios[:,None, None]
                    # reference_points_input [bz, nf, 300, 4, 2]


            output, output_box, sampling_locations, attention_weights, reference_out, mem_query, mem_query_pos = \
                layer(output, output_box, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index,
                      src_padding_mask, mem, mem_pos, mem_token_mask, mem_query, mem_query_pos, lid=lid) # alter here last layer false as its fina
            if self.dec_ref_prop:
                attn_weights_all.append(attention_weights)
            if self.analysis or self.dec_ref_prop:
                sampling_locations_all.append(sampling_locations)
            if self.init_ref_after_memory:
                # if isinstance(reference_points, tuple):
                #     reference_points = reference_points[0]
                if lid == 0:
                    init_reference_out = reference_out
                    if reference_out is not None:
                        reference_points = reference_out

            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output if self.online else output_box)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

                if inter_references is not None:
                    reference_points = (inter_references[lid] * reference_points).sigmoid()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_box.append(output_box)
                intermediate_reference_points.append(reference_points)
                # intermediate_samples.append(samples_keep)
            if self.online:
                output = output.squeeze(1)

        if self.dec_ref_prop:
            attn_weights_all = torch.stack(attn_weights_all, dim=1)
        if self.analysis or self.dec_ref_prop:
            sampling_locations_all = torch.stack(sampling_locations_all, dim=1)
        if self.return_intermediate:
            return torch.stack(intermediate), None if self.online else torch.stack(intermediate_box), \
                   torch.stack(intermediate_reference_points), None, init_reference_out, mem_query, mem_query_pos,\
                   [sampling_locations_all, attn_weights_all] if self.dec_ref_prop else None, sampling_locations_all if self.analysis else None

        return output, reference_points


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_deforamble_transformer(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_frames=1 if args.online else args.num_frames,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        propagate_ref_points=args.propagate_ref_points,
        propagate_time_ref= args.propagate_time_ref,
        memory_frames= args.memory_support,
        num_classes = args.num_classes,
        last_ref = args.last_ref,
        alternate_attn=args.alternate_attn,
        attn_at=args.attn_at,
        init_ref_after_memory=args.init_ref_after_memory,
        enc_ref_prop = args.enc_ref_prop,
        num_queries=args.num_queries,
        mask_query = args.mask_query,
        memory_as_query=args.memory_as_query,
        dec_ref_prop = args.dec_ref_prop,
        dec_ref_keep_ratio=args.dec_ref_keep_ratio,
        enc_aux_loss=args.enc_aux_loss,
        train=args.train,
        analysis=args.analysis,
        propagate_inter_ref_2=args.propagate_inter_ref_2,
        propagate_ref_additive=args.propagate_ref_additive)





