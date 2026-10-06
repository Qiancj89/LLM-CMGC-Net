"""Matched bag aggregators for the selected LLM-CMGC-Net architecture."""

from pathlib import Path
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vl_cmgc_ot_net import OptimalTransportMIL, VL_CMGC_Net


class MatchedBagFusion(OptimalTransportMIL):
    """Keep the original projection/fusion layers and vary only bag assignment."""

    def __init__(self, dim=768, mode="sinkhorn", epsilon=0.1, iters=10):
        super().__init__(dim=dim, epsilon=epsilon, iters=iters)
        if mode not in {"mean", "attention", "sinkhorn"}:
            raise ValueError(f"Unknown aggregation: {mode}")
        self.mode = mode

    def forward(self, bag_b, feat_c, valid_mask):
        if self.mode == "sinkhorn":
            return super().forward(bag_b, feat_c, valid_mask)

        with torch.amp.autocast("cuda", enabled=False):
            bag = bag_b.float()
            ceus = feat_c.float()
            valid = valid_mask.bool()
            if not valid.any(dim=1).all():
                raise ValueError("Each examination needs at least one valid B-mode view")
            batch, channels, height, width = ceus.shape
            positions = height * width

            if self.mode == "mean":
                weight = valid.float() / valid.sum(dim=1, keepdim=True)
                weight = weight.unsqueeze(-1).expand(-1, -1, positions)
            else:
                query = ceus.flatten(2).transpose(1, 2)
                key = F.normalize(self.proj_b(bag), dim=-1)
                query = F.normalize(self.proj_c(query), dim=-1)
                score = torch.bmm(key, query.transpose(1, 2)) / self.epsilon
                score = score.masked_fill(~valid.unsqueeze(-1), torch.finfo(score.dtype).min)
                weight = torch.softmax(score, dim=1)

            transported = torch.bmm(weight.transpose(1, 2), bag)
            transported = transported.transpose(1, 2).reshape(batch, channels, height, width)
            fused = self.fusion_conv(torch.cat((ceus, transported), dim=1))
            plan = weight / positions
        return fused.to(feat_c.dtype), plan


def build_model(mode, text_model_name):
    model = VL_CMGC_Net(
        num_classes_list=[4, 2, 2, 2, 2, 2, 2],
        text_model_name=text_model_name,
    )
    model.ot_mil = MatchedBagFusion(dim=768, mode=mode)
    return model
