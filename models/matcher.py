# ------------------------------------------------------------------------
# InstanceFormer Sequence Matching
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
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, generalized_multi_box_iou


class HungarianMatcher(nn.Module):
    def __init__(self,
                 multi_frame: bool,
                 cost_class: float = 1,
                 cost_bbox: float = 1,
                 cost_giou: float = 1,
                 cost_mask: float = 1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.multi_frame = multi_frame
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_mask = cost_mask
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0 or cost_mask != 0, "all costs cant be 0"

    def forward(self, outputs, targets, nf, valid_ratios, online=False, enc_aux_loss=False):
        with torch.no_grad():
            if enc_aux_loss:
                bs, num_queries = outputs["pred_logits"].shape[0], outputs["pred_logits"].shape[2]
            else:
                bs, num_queries = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            if enc_aux_loss:
                out_prob = outputs["pred_logits"].flatten(0, 1).flatten(0, 1).sigmoid()
            else:
                out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            # Also concat the target labels and boxes
            tgt_ids = torch.cat([v["labels"] for v in targets])
            if online:
                tgt_bbox = torch.cat([v["boxes"][nf] for v in targets])  # take current frame boxes
                # if not self.multi_frame:
                #     tgt_bbox = tgt_bbox[None, ...]
                out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

                cost_bbox = torch.cdist(out_bbox, tgt_bbox)
                cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
            else:
                if enc_aux_loss:
                    tgt_bbox = torch.cat([v["boxes"][1:] for v in targets],1)
                else:
                    tgt_bbox = torch.cat([v["boxes"] for v in targets])

                out_bbox = outputs["pred_boxes"].permute(0, 2, 1, 3).flatten(0, 1)  # [batch_size * num_queries,nf, 4]
                num_insts = len(tgt_ids)
                tgt_bbox = tgt_bbox.reshape(num_insts, nf, 4)
                cost_bbox = torch.cdist(out_bbox.flatten(1, 2), tgt_bbox.flatten(1, 2))
                cost_giou = 0
                for i in range(nf):
                    cost_giou += -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox[:, i]),
                                                      box_cxcywh_to_xyxy(tgt_bbox[:, i]))
                cost_giou = cost_giou / nf

            # Compute the classification cost.
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
            # print('pos_cost_class', pos_cost_class.shape)
            # print('tgt_ids', tgt_ids)
            # Final cost matrix
            if enc_aux_loss:
                cost_class = cost_class.view(nf, -1, num_insts).sum(0) / nf
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            # C = C.view(bs, num_queries, -1).cpu()
            C = C.view(bs, num_queries, -1).data.cpu()
            sizes = [len(v["labels"]) for v in targets]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    # output single frame, multi frame
    return HungarianMatcher(multi_frame= True, #args.num_frames > 1, #True, # True, False
                            cost_class=args.set_cost_class,
                            cost_bbox=args.set_cost_bbox,
                            cost_giou=args.set_cost_giou)


