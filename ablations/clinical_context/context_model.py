import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CODE_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("LLM_CMGC_DATA_ROOT", CODE_ROOT)).resolve()
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from vl_cmgc_ot_net import VL_CMGC_Net


class ContextAblationNet(VL_CMGC_Net):
    """LLM-CMGC-Net with controlled clinical-context alternatives."""

    def __init__(self, context_mode, tabular_dim=25, **kwargs):
        if context_mode not in {"no_text", "tabular"}:
            raise ValueError("context_mode must be 'no_text' or 'tabular'")
        super().__init__(**kwargs)
        self.context_mode = context_mode
        self.text_encoder = nn.Identity()

        if context_mode == "tabular":
            self.tabular_encoder = nn.Sequential(
                nn.Linear(tabular_dim, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(256, 768),
            )
        else:
            self.register_buffer("null_context", torch.zeros(1, 768))

    def _context_features(self, batch_size, device, tabular):
        if self.context_mode == "tabular":
            if tabular is None:
                raise ValueError("tabular context is required in tabular mode")
            return self.tabular_encoder(tabular)
        return self.null_context.expand(batch_size, -1).to(device)

    def forward(self, pb, pc, ubs, uv, tabular=None):
        batch_size = pb.size(0)
        context_feat = self._context_features(batch_size, pb.device, tabular)
        feat_c = self.encoder(pc)
        feat_pb = self.encoder(pb)

        bag_features = [F.adaptive_avg_pool2d(feat_pb, (1, 1)).flatten(1)]
        valid_mask = [torch.ones(batch_size, dtype=torch.bool, device=pb.device)]
        unpaired_spatial_feats = []
        for index in range(ubs.size(1)):
            valid = uv[:, index] == 1.0
            feat_u = self.encoder(ubs[:, index])
            unpaired_spatial_feats.append(feat_u)
            pooled = F.adaptive_avg_pool2d(feat_u, (1, 1)).flatten(1)
            padded = torch.zeros_like(pooled)
            if valid.any():
                padded[valid] = pooled[valid]
            bag_features.append(padded)
            valid_mask.append(valid)

        bag_b = torch.stack(bag_features, dim=1)
        valid_mask = torch.stack(valid_mask, dim=1)
        fused_spatial, _ = self.ot_mil(bag_b, feat_c, valid_mask)

        seg_out_main = self.decoder_main(fused_spatial)
        seg_out_pb_aux = self.decoder_pb_aux(feat_pb)
        aux_stack = torch.stack(unpaired_spatial_feats, dim=1)
        b, n, c, h, w = aux_stack.shape
        aux_out = self.decoder_bag(aux_stack.reshape(b * n, c, h, w))
        seg_out_bag = aux_out.reshape(b, n, 1, aux_out.size(2), aux_out.size(3))

        patient_feat = F.adaptive_avg_pool2d(fused_spatial, (1, 1)).flatten(1)
        if self.context_mode == "tabular":
            image_norm = F.normalize(patient_feat, dim=-1)
            context_norm = F.normalize(context_feat, dim=-1)
            logits = self.logit_scale.exp() * image_norm @ context_norm.t()
            target = torch.arange(batch_size, device=pb.device)
            context_loss = (
                F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target)
            ) / 2.0
        else:
            context_loss = patient_feat.new_zeros(())

        node_features = []
        concept_predictions = []
        for index in range(self.num_concepts):
            node = self.concept_generators[index](patient_feat)
            concept_predictions.append(self.concept_classifiers[index](node))
            node_features.append(node)
        node_features.append(self.global_generator(patient_feat))
        node_features.append(self.text_proj(context_feat))

        graph_nodes = torch.stack(node_features, dim=1)
        adjacency = F.softmax(self.adj_matrix.expand(batch_size, -1, -1), dim=-1)
        graph_output = self.gcn2(self.gcn1(graph_nodes, adjacency), adjacency)
        malignancy = self.final_cls(graph_output.reshape(batch_size, -1))
        return (
            seg_out_main,
            seg_out_bag,
            seg_out_pb_aux,
            context_loss,
            concept_predictions,
            malignancy,
        )
