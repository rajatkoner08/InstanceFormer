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
from torch.nn.init import xavier_uniform_, constant_, normal_
from util.misc import inverse_sigmoid
from models.ops.modules.ms_deform_attn import MSDeformAttn


class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024,
                 dropout=0.1, activation="relu", return_intermediate_dec=False, num_frames=1, num_feature_levels=4,
                 dec_n_points=4, enc_n_points=4, memory_frames=False, memory_token = False, num_classes=42, propagate_ref_points=False,
                 num_queries=300, train=True, propagate_ref_additive=False, analysis=False):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.num_feature_levels = num_feature_levels
        self.propagate_ref_points = propagate_ref_points
        self.mframe = memory_frames
        self.mtoken = memory_token
        self.num_queries = num_queries
        self.training = train
        self.analysis = analysis
        if self.mframe > 0 and self.mtoken > 0:
            self.mem_token_proj = nn.Linear(d_model + 4 + num_classes, d_model)
            self.mem_token_proj_drp = nn.Dropout(dropout)

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels,
                                                          nhead, enc_n_points)

        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)
        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels,
                                                          nhead, dec_n_points, memory=memory_frames,
                                                          token= self.mtoken, num_queries=self.num_queries)

        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, num_frames,
                                                    return_intermediate_dec, self.analysis)
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self.reference_points = nn.Linear(d_model, 2)
        self.num_feature_levels = num_feature_levels
        self.dec_n_points = dec_n_points
        self.num_queries = num_queries
        self.num_decoder_layers = num_decoder_layers
        self.propagate_ref_additive=propagate_ref_additive
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

    def forward(self, srcs, masks, pos_embeds, query_embed=None, tgt=None, init_reference=None, mem_token=None,
                mem_token_pos=None, mem_token_mask=None):
        assert query_embed is not None
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

            src = src.flatten(3).transpose(2, 3)  # src: [N, nf, Hi*Wi, C]
            mask = mask.flatten(2)  # mask: [N, nf, Hi*Wi]
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
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m[:, 0]) for m in masks], 1)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        # prepare input for decoder
        bs, nf, _, c = memory.shape
        reference_points = self.init_referance(init_reference, query_embed, self.propagate_ref_additive)

        if not isinstance(reference_points, tuple):
            reference_points = reference_points.unsqueeze(1).repeat(1, nf, 1, 1)  # [bz,nf,300,2]
            init_reference_out = reference_points

        # decoder
        hs, inter_references, inter_samples, reference_out, sampling_locations_all = self.decoder(tgt, reference_points, memory,
                                                                                  spatial_shapes, level_start_index,
                                                                                  valid_ratios, query_embed,
                                                                                  mask_flatten, mem_token,
                                                                                  mem_token_pos, mem_token_mask)

        return hs, memory, init_reference_out, inter_references, inter_samples, None, valid_ratios, level_start_index, sampling_locations_all


    def init_referance(self, init_reference, query_embed, propagate_ref_additive=False):
        if init_reference is not None:
            if propagate_ref_additive:
                reference_points = (init_reference[:, 0, ...] + self.reference_points(query_embed).sigmoid()).sigmoid()
            else:
                reference_points = (init_reference[:, 0, ...] * self.reference_points(query_embed).sigmoid()).sigmoid()
        else:
            reference_points = self.reference_points(query_embed).sigmoid()
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

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2, sampling_locations = self.self_attn(self.with_pos_embed(src, pos), None, reference_points, src,spatial_shapes,
                                                  level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # ffn
        src = self.forward_ffn(src)
        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.bbox_embed = None
        self.class_embed = None

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

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation="relu", n_levels=4, n_heads=8, n_points=4,
                 memory=0, token =0, num_queries=300):
        super().__init__()
        self.num_queries = num_queries
        self.query_mask = torch.zeros(num_queries, dtype=bool, requires_grad=False)
        self.num_mask = int(num_queries * dropout / 3)
        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points, 'decode')
        self.dropout1 = nn.Dropout(dropout)
        self.dropout1_box = nn.Dropout(dropout)
        self.norm1_box = nn.LayerNorm(d_model)
        # self attention for mask&class query
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        if memory > 0 and token > 0:
            self.cross_attn_mem = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
            self.dropout2_mem = nn.Dropout(dropout)
            self.norm2_mem = nn.LayerNorm(d_model)
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

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index,
                src_padding_mask=None, mem=None, mem_pos=None, mem_token_mask=None, lid=0):

        q1 = k1 = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q1.transpose(0, 1), k1.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        init_reference_out = None
        # cross attention
        tgt2, sampling_locations, attention_weights = self.cross_attn(tgt,self.with_pos_embed(tgt,query_pos),
                                                                         reference_points,src, src_spatial_shapes,
                                                                         level_start_index, src_padding_mask)
        tgt2 = tgt2.squeeze(1)
        tgt = tgt + self.dropout1_box(tgt2)
        tgt = self.norm1_box(tgt)

        # memory attn
        if mem !=None:
            tgt = self.memory_attn(mem, mem_pos, mem_token_mask, query_pos, tgt)

        tgt = self.forward_ffn_box(tgt)
        tgt = tgt.unsqueeze(1)  # inflate dimesion to frame

        return tgt, sampling_locations, attention_weights, init_reference_out

    def memory_attn(self, mem, mem_pos, mem_token_mask, query_pos, tgt):
        tgt2 = self.cross_attn_mem(query=self.with_pos_embed(tgt, query_pos).permute(1, 0, 2),
                                   key=self.with_pos_embed(mem, mem_pos),
                                   value=mem, attn_mask=None, key_padding_mask=mem_token_mask)[
            0]  # todo replaces the key padding mask, and exp wd pos
        tgt = tgt + self.dropout2_mem(tgt2.permute(1, 0, 2))
        tgt = self.norm2_mem(tgt)
        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, num_frames, return_intermediate=False, analysis=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.bbox_embed = None
        self.class_embed = None
        self.num_frames = num_frames
        self.analysis = analysis

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None,
                mem=None, mem_pos=None, mem_token_mask=None):
        output = tgt
        intermediate = []
        intermediate_reference_points = []
        init_reference_out = None
        sampling_locations_all = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, :, None] * torch.cat(
                    [src_valid_ratios, src_valid_ratios], -1)[:, None, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, :, None] * src_valid_ratios[:, None,
                                                                           None]  # reference_points_input [bz, nf, 300, 4, 2]

            output, sampling_locations, attention_weights, reference_out = \
                layer(output, query_pos, reference_points_input, src, src_spatial_shapes,
                      src_level_start_index,
                      src_padding_mask, mem, mem_pos, mem_token_mask,
                      lid=lid)  # alter here last layer false as its fina
            if self.analysis:
                sampling_locations_all.append(sampling_locations)
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
            output = output.squeeze(1)

        if self.analysis :
            sampling_locations_all = torch.stack(sampling_locations_all, dim=1)
        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points), None, init_reference_out, \
                sampling_locations_all if self.analysis else None

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
        num_frames=1,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        propagate_ref_points=args.propagate_ref_points,
        memory_frames=args.memory_support,
        memory_token=args.memory_token,
        num_classes=args.num_classes,
        num_queries=args.num_queries,
        train=args.train,
        propagate_ref_additive=args.propagate_ref_additive,
        analysis=args.analysis)
