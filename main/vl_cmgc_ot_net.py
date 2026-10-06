import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from transformers import AutoModel


class GCNLayer(nn.Module):
    """Graph convolution over clinical-indicator concepts."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x, adj):
        support = self.fc(x)
        output = torch.bmm(adj, support)
        return F.relu(output)


class OptimalTransportMIL(nn.Module):
    """Align multiple image instances using numerically stable transport."""
    def __init__(self, dim, epsilon=0.1, iters=10):
        super().__init__()
        self.epsilon = epsilon
        self.iters = iters
        self.proj_b = nn.Sequential(nn.Linear(dim, dim//2), nn.GELU())
        self.proj_c = nn.Sequential(nn.Linear(dim, dim//2), nn.GELU())
        self.fusion_conv = nn.Conv2d(dim * 2, dim, kernel_size=1)

    def forward(self, bag_b, feat_c, valid_mask):
        with torch.amp.autocast('cuda', enabled=False):
            bag_b_32, feat_c_32 = bag_b.float(), feat_c.float()
            B, D, H, W = feat_c_32.shape
            M = H * W

            feat_c_flat = feat_c_32.view(B, D, M).transpose(1, 2)
            b_proj = F.normalize(self.proj_b(bag_b_32), dim=-1)
            c_proj = F.normalize(self.proj_c(feat_c_flat), dim=-1)

            cost = 1.0 - torch.bmm(b_proj, c_proj.transpose(1, 2))
            # Padded bag instances receive no transport mass.
            cost = cost.masked_fill(valid_mask.unsqueeze(-1) == 0, 1e4)

            K = torch.exp(-cost / self.epsilon)
            mu = valid_mask.float() / (valid_mask.float().sum(dim=1, keepdim=True) + 1e-8)
            nu = torch.ones(B, M, device=cost.device) / M
            u, v = torch.ones_like(mu), torch.ones_like(nu)

            for _ in range(self.iters):
                u = mu / (torch.bmm(K, v.unsqueeze(-1)).squeeze(-1) + 1e-8)
                v = nu / (torch.bmm(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1) + 1e-8)

            P = u.unsqueeze(-1) * K * v.unsqueeze(1)
            transported_b = torch.bmm((P * M).transpose(1, 2), bag_b_32).transpose(1, 2).view(B, D, H, W)
            fused_spatial = self.fusion_conv(torch.cat([feat_c_32, transported_b], dim=1))

        return fused_spatial.type_as(feat_c), P


class SimpleDecoder(nn.Module):
    """Upsampling decoder for a dense segmentation output."""
    def __init__(self, dim=768):
        super().__init__()
        self.up_conv1 = nn.ConvTranspose2d(dim, 384, 2, 2)
        self.up_conv2 = nn.ConvTranspose2d(384, 192, 2, 2)
        self.up_conv3 = nn.ConvTranspose2d(192, 96, 2, 2)
        self.up_conv4 = nn.ConvTranspose2d(96, 32, 4, 4)
        self.seg_head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        return self.seg_head(self.up_conv4(self.up_conv3(self.up_conv2(self.up_conv1(x)))))


class VL_CMGC_Net(nn.Module):
    def __init__(self, num_classes_list=[4, 2, 2, 2, 2, 2, 2], text_model_name="hfl/chinese-macbert-base"):
        super().__init__()
        self.encoder = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT).features
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.ot_mil = OptimalTransportMIL(dim=768)

        self.decoder_main = SimpleDecoder(dim=768)
        self.decoder_bag = SimpleDecoder(dim=768)
        self.decoder_pb_aux = SimpleDecoder(dim=768)

        self.num_concepts = len(num_classes_list)
        self.num_nodes = self.num_concepts + 2

        self.concept_generators = nn.ModuleList([
            nn.Sequential(nn.Dropout(p=0.5), nn.Linear(768, 128)) for _ in range(self.num_concepts)
        ])

        self.global_generator = nn.Linear(768, 128)
        self.text_proj = nn.Linear(768, 128)
        self.concept_classifiers = nn.ModuleList([nn.Linear(128, num_c) for num_c in num_classes_list])

        self.adj_matrix = nn.Parameter(torch.rand(1, self.num_nodes, self.num_nodes))
        self.gcn1 = GCNLayer(128, 64)
        self.gcn2 = GCNLayer(64, 32)
        self.final_cls = nn.Linear(32 * self.num_nodes, 1)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, pb, pc, ubs, uv, input_ids, attention_mask):
        B = pb.size(0)

        text_feat = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        feat_c = self.encoder(pc)
        feat_pb = self.encoder(pb)

        bag_features = [F.adaptive_avg_pool2d(feat_pb, (1, 1)).squeeze(-1).squeeze(-1)]
        valid_mask = [torch.ones(B, dtype=torch.bool, device=pb.device)]
        unpaired_spatial_feats = []

        for i in range(ubs.size(1)):
            v_idx = (uv[:, i] == 1.0)
            feat_u = self.encoder(ubs[:, i])
            unpaired_spatial_feats.append(feat_u)

            pooled_u = F.adaptive_avg_pool2d(feat_u, (1, 1)).squeeze(-1).squeeze(-1)
            padded_feat = torch.zeros_like(pooled_u)
            if v_idx.any(): padded_feat[v_idx] = pooled_u[v_idx]

            bag_features.append(padded_feat)
            valid_mask.append(v_idx)

        bag_b = torch.stack(bag_features, dim=1)
        valid_mask = torch.stack(valid_mask, dim=1)

        fused_spatial, _ = self.ot_mil(bag_b, feat_c, valid_mask)

        seg_out_main = self.decoder_main(fused_spatial)
        # This auxiliary decoder supervises paired B-mode features before OT fusion.
        seg_out_pb_aux = self.decoder_pb_aux(feat_pb)

        aux_stack = torch.stack(unpaired_spatial_feats, dim=1)
        _, N_aux, C, H_f, W_f = aux_stack.size()
        aux_flat = aux_stack.reshape(B * N_aux, C, H_f, W_f)
        aux_out_flat = self.decoder_bag(aux_flat)
        seg_out_bag = aux_out_flat.reshape(B, N_aux, 1, aux_out_flat.size(2), aux_out_flat.size(3))

        patient_global_feat = F.adaptive_avg_pool2d(fused_spatial, (1, 1)).squeeze(-1).squeeze(-1)

        img_norm = F.normalize(patient_global_feat, dim=-1)
        txt_norm = F.normalize(text_feat, dim=-1)
        logits_per_image = self.logit_scale.exp() * img_norm @ txt_norm.t()
        loss_itc = (F.cross_entropy(logits_per_image, torch.arange(B, device=pb.device)) +
                    F.cross_entropy(logits_per_image.t(), torch.arange(B, device=pb.device))) / 2.0

        node_features = []
        concept_preds = []
        for i in range(self.num_concepts):
            node_feat = self.concept_generators[i](patient_global_feat)
            concept_preds.append(self.concept_classifiers[i](node_feat))
            node_features.append(node_feat)

        node_features.append(self.global_generator(patient_global_feat))
        node_features.append(self.text_proj(text_feat))

        graph_nodes = torch.stack(node_features, dim=1)
        adj_norm = F.softmax(self.adj_matrix.expand(B, -1, -1), dim=-1)
        gcn_out = self.gcn2(self.gcn1(graph_nodes, adj_norm), adj_norm)

        malignancy_pred = self.final_cls(gcn_out.view(B, -1))

        return seg_out_main, seg_out_bag, seg_out_pb_aux, loss_itc, concept_preds, malignancy_pred
