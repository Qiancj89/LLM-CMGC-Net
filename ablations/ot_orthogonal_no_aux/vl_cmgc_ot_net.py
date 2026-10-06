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
    """Align multiple image instances using optimal transport."""
    def __init__(self, dim, epsilon=0.1, iters=10):
        super().__init__()
        self.epsilon = epsilon
        self.iters = iters

        self.proj_b = nn.Sequential(nn.Linear(dim, dim//2), nn.GELU())
        self.proj_c = nn.Sequential(nn.Linear(dim, dim//2), nn.GELU())
        self.fusion_conv = nn.Conv2d(dim * 2, dim, kernel_size=1)

    def forward(self, bag_b, feat_c, valid_mask):

        with torch.amp.autocast('cuda', enabled=False):

            bag_b_32 = bag_b.float()
            feat_c_32 = feat_c.float()

            B, D, H, W = feat_c_32.shape
            M = H * W

            feat_c_flat = feat_c_32.view(B, D, M).transpose(1, 2)
            b_proj = F.normalize(self.proj_b(bag_b_32), dim=-1)
            c_proj = F.normalize(self.proj_c(feat_c_flat), dim=-1)

            cost = 1.0 - torch.bmm(b_proj, c_proj.transpose(1, 2))

            cost = cost.masked_fill(valid_mask.unsqueeze(-1) == 0, 1e4)
            K = torch.exp(-cost / self.epsilon)

            mu = valid_mask.float() / (valid_mask.float().sum(dim=1, keepdim=True) + 1e-8)
            nu = torch.ones(B, M, device=cost.device) / M

            u, v = torch.ones_like(mu), torch.ones_like(nu)

            for _ in range(self.iters):
                u = mu / (torch.bmm(K, v.unsqueeze(-1)).squeeze(-1) + 1e-8)
                v = nu / (torch.bmm(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1) + 1e-8)

            P = u.unsqueeze(-1) * K * v.unsqueeze(1)
            P_scaled = P * M

            transported_b = torch.bmm(P_scaled.transpose(1, 2), bag_b_32)
            transported_b = transported_b.transpose(1, 2).view(B, D, H, W)
            fused_spatial = self.fusion_conv(torch.cat([feat_c_32, transported_b], dim=1))

        return fused_spatial.type_as(feat_c), P


class OrthogonalConceptGenerator(nn.Module):
    def __init__(self, in_dim, concept_dim, num_concepts):
        super().__init__()
        self.generators = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(in_dim, concept_dim)
            ) for _ in range(num_concepts)])
    def forward(self, x): return [gen(x) for gen in self.generators]


class VL_CMGC_Net(nn.Module):
    def __init__(self, num_classes_list=[4, 2, 2, 2, 2, 2, 2], text_model_name="hfl/chinese-macbert-base"):
        super().__init__()

        self.encoder = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT).features
        self.text_encoder = AutoModel.from_pretrained(text_model_name)

        self.ot_mil = OptimalTransportMIL(dim=768)

        self.up_conv1 = nn.ConvTranspose2d(768, 384, 2, 2)
        self.up_conv2 = nn.ConvTranspose2d(384, 192, 2, 2)
        self.up_conv3 = nn.ConvTranspose2d(192, 96, 2, 2)
        self.up_conv4 = nn.ConvTranspose2d(96, 32, 4, 4)
        self.seg_head = nn.Conv2d(32, 1, 1)

        self.aux_up_conv1 = nn.ConvTranspose2d(768, 384, 2, 2)
        self.aux_up_conv2 = nn.ConvTranspose2d(384, 192, 2, 2)
        self.aux_up_conv3 = nn.ConvTranspose2d(192, 96, 2, 2)
        self.aux_up_conv4 = nn.ConvTranspose2d(96, 32, 4, 4)
        self.aux_seg_head = nn.Conv2d(32, 1, 1)

        self.num_concepts = len(num_classes_list)
        self.num_nodes = self.num_concepts + 2
        self.adj_matrix = nn.Parameter(torch.rand(1, self.num_nodes, self.num_nodes))

        self.concept_gen = OrthogonalConceptGenerator(768, 128, self.num_concepts)
        self.global_gen = nn.Linear(768, 128)
        self.text_proj = nn.Linear(768, 128)

        self.concept_classifiers = nn.ModuleList([nn.Linear(128, num_c) for num_c in num_classes_list])

        self.gcn1 = GCNLayer(128, 64)
        self.gcn2 = GCNLayer(64, 32)
        self.final_cls = nn.Linear(32 * self.num_nodes, 1)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, pb, pc, ubs, uv, input_ids, attention_mask):
        B = pb.size(0)

        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = text_out.pooler_output

        feat_c = self.encoder(pc)
        feat_pb = self.encoder(pb)
        pooled_pb = F.adaptive_avg_pool2d(feat_pb, (1, 1)).squeeze(-1).squeeze(-1)

        bag_features = [pooled_pb]
        valid_mask = [torch.ones(B, dtype=torch.bool, device=pb.device)]
        unpaired_spatial_feats = []

        N_max = ubs.size(1)
        for i in range(N_max):
            v_idx = (uv[:, i] == 1.0)
            u_b = ubs[:, i]
            feat_u = self.encoder(u_b)
            unpaired_spatial_feats.append(feat_u)

            pooled_u = F.adaptive_avg_pool2d(feat_u, (1, 1)).squeeze(-1).squeeze(-1)
            padded_feat = torch.zeros_like(pooled_u)
            if v_idx.any(): padded_feat[v_idx] = pooled_u[v_idx]

            bag_features.append(padded_feat)
            valid_mask.append(v_idx)

        bag_b = torch.stack(bag_features, dim=1)
        valid_mask = torch.stack(valid_mask, dim=1)

        fused_spatial, _ = self.ot_mil(bag_b, feat_c, valid_mask)

        seg_out_main = self.seg_head(self.up_conv4(self.up_conv3(self.up_conv2(self.up_conv1(fused_spatial)))))

        aux_stack = torch.stack(unpaired_spatial_feats, dim=1)
        _, N_aux, C, H_f, W_f = aux_stack.size()
        aux_flat = aux_stack.reshape(B * N_aux, C, H_f, W_f)
        aux_out_flat = self.aux_seg_head(self.aux_up_conv4(self.aux_up_conv3(self.aux_up_conv2(self.aux_up_conv1(aux_flat)))))
        seg_out_aux = aux_out_flat.reshape(B, N_aux, 1, aux_out_flat.size(2), aux_out_flat.size(3))

        patient_global_feat = F.adaptive_avg_pool2d(fused_spatial, (1, 1)).squeeze(-1).squeeze(-1)

        img_norm = F.normalize(patient_global_feat, dim=-1)
        txt_norm = F.normalize(text_feat, dim=-1)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * img_norm @ txt_norm.t()
        loss_itc = (F.cross_entropy(logits_per_image, torch.arange(B, device=pb.device)) + F.cross_entropy(logits_per_image.t(), torch.arange(B, device=pb.device))) / 2.0

        concepts = self.concept_gen(patient_global_feat)
        concept_preds = [self.concept_classifiers[i](concepts[i]) for i in range(self.num_concepts)]

        global_node = self.global_gen(patient_global_feat)
        text_node = self.text_proj(text_feat)

        graph_nodes = torch.stack(concepts + [global_node, text_node], dim=1)
        adj_norm = F.softmax(self.adj_matrix.expand(B, -1, -1), dim=-1)
        gcn_out = self.gcn2(self.gcn1(graph_nodes, adj_norm), adj_norm)

        malignancy_pred = self.final_cls(gcn_out.view(B, -1))

        ortho_loss = 0.0
        if self.training:
            c_tensor = F.normalize(torch.stack(concepts, dim=1), dim=-1)
            sim = torch.bmm(c_tensor, c_tensor.transpose(1, 2))
            identity = torch.eye(self.num_concepts, device=pc.device).unsqueeze(0)
            ortho_loss = torch.sum((sim - identity) ** 2) / B

        return seg_out_main, seg_out_aux, ortho_loss, loss_itc, concept_preds, malignancy_pred
