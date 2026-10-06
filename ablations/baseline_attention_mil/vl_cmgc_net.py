import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from transformers import AutoModel
import numpy as np


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(x_cat))


class SpatialAlignedOT(nn.Module):
    """Align B-mode and CEUS spatial features using optimal transport."""
    def __init__(self, dim, epsilon=0.1, iters=5):
        super().__init__()
        self.epsilon = epsilon
        self.iters = iters
        self.proj_q = nn.Conv2d(dim, dim // 4, 1)
        self.proj_k = nn.Conv2d(dim, dim // 4, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, feat_b, feat_c):

        with torch.amp.autocast('cuda', enabled=False):
            fb32, fc32 = feat_b.float(), feat_c.float()
            B, C, H, W = fc32.shape
            M = H * W

            q = self.proj_q(fc32).view(B, -1, M).permute(0, 2, 1)
            k = self.proj_k(fb32).view(B, -1, M)

            dist = 1.0 - torch.bmm(F.normalize(q, dim=-1), F.normalize(k, dim=1))
            K = torch.exp(-dist / self.epsilon)

            u = torch.ones(B, M, device=fc32.device) / M
            v = torch.ones(B, M, device=fc32.device) / M

            for _ in range(self.iters):
                u = (1.0 / M) / (torch.bmm(K, v.unsqueeze(2)).squeeze(2) + 1e-8)
                v = (1.0 / M) / (torch.bmm(K.transpose(1, 2), u.unsqueeze(2)).squeeze(2) + 1e-8)

            T = u.unsqueeze(2) * K * v.unsqueeze(1) * M

            feat_b_flat = fb32.view(B, C, M).permute(0, 2, 1)
            aligned_feat = torch.bmm(T, feat_b_flat).permute(0, 2, 1).view(B, C, H, W)

            return (fc32 + self.gamma * aligned_feat).to(feat_b.dtype)


class OrthogonalConceptGenerator(nn.Module):
    def __init__(self, in_dim, concept_dim, num_concepts):
        super().__init__()
        self.num_concepts = num_concepts
        self.generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, concept_dim),
                nn.ReLU(),
                nn.LayerNorm(concept_dim)
            ) for _ in range(num_concepts)
        ])

    def forward(self, x):
        return [gen(x) for gen in self.generators]


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        support = torch.matmul(x, self.weight)
        output = torch.bmm(adj, support)
        return F.relu(output)


class TextGuidedDecoderBlock(nn.Module):
    """Decoder block with residual, spatial-attention, and FiLM modulation."""
    def __init__(self, in_channels, out_channels, text_dim=768):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(out_channels)
        self.act = nn.GELU()
        self.sa = SpatialAttention()

        self.film_gamma = nn.Linear(text_dim, out_channels)
        self.film_beta = nn.Linear(text_dim, out_channels)

        nn.init.zeros_(self.film_gamma.weight)
        nn.init.zeros_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)

    def forward(self, x, text_feat):
        x = self.up(x)
        res = x

        feat = self.act(self.norm1(self.conv1(x)))
        feat = self.norm2(self.conv2(feat))

        gamma = self.film_gamma(text_feat).unsqueeze(2).unsqueeze(3)
        beta = self.film_beta(text_feat).unsqueeze(2).unsqueeze(3)
        feat = feat * (1 + gamma) + beta

        feat = self.sa(feat)
        return self.act(feat + res)


class TextEnhancedOTDecoder(nn.Module):
    def __init__(self, dim=768, text_dim=768):
        super().__init__()
        self.up_conv1 = TextGuidedDecoderBlock(dim, dim//2, text_dim)
        self.up_conv2 = TextGuidedDecoderBlock(dim//2, dim//4, text_dim)
        self.up_conv3 = TextGuidedDecoderBlock(dim//4, dim//8, text_dim)
        self.up_conv4 = TextGuidedDecoderBlock(dim//8, 64, text_dim)
        self.up_conv5 = TextGuidedDecoderBlock(64, 32, text_dim)
        self.seg_head = nn.Conv2d(32, 1, 1)

    def forward(self, x, text_feat):
        x = self.up_conv1(x, text_feat)
        x = self.up_conv2(x, text_feat)
        x = self.up_conv3(x, text_feat)
        x = self.up_conv4(x, text_feat)
        x = self.up_conv5(x, text_feat)
        return self.seg_head(x)


class VL_CMGC_Net(nn.Module):
    def __init__(self, num_classes_list=[4, 2, 2, 2, 2, 2, 2], concept_dim=128, text_model_name="hfl/chinese-macbert-base"):
        super().__init__()

        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        self.enc_b = convnext_tiny(weights=weights).features
        self.enc_c = convnext_tiny(weights=weights).features
        self.sa_ot = SpatialAlignedOT(dim=768)

        self.text_encoder = AutoModel.from_pretrained(text_model_name)

        for param in self.text_encoder.embeddings.parameters(): param.requires_grad = False
        for param in self.text_encoder.encoder.layer[:6].parameters(): param.requires_grad = False

        self.num_concepts = len(num_classes_list)
        self.concept_gen = OrthogonalConceptGenerator(768*2, concept_dim, self.num_concepts)
        self.global_img_gen = nn.Linear(768*2, concept_dim)

        self.text_node_gen = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, concept_dim)
        )

        self.num_nodes = self.num_concepts + 2
        self.adj_matrix = nn.Parameter(torch.rand(1, self.num_nodes, self.num_nodes))
        self.gcn1 = GCNLayer(concept_dim, 64)
        self.gcn2 = GCNLayer(64, 32)

        self.mal_head = nn.Sequential(
            nn.Linear(32 * self.num_nodes, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
        self.concept_classifiers = nn.ModuleList([nn.Linear(concept_dim, nc) for nc in num_classes_list])
        self.mil_attn = nn.Sequential(nn.Linear(768, 128), nn.Tanh(), nn.Linear(128, 1))

        self.decoder_main = TextEnhancedOTDecoder(dim=768, text_dim=768)
        self.decoder_aux = TextEnhancedOTDecoder(dim=768, text_dim=768)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, pb, pc, ubs, uv, input_ids, attention_mask):
        B = pc.shape[0]
        N = ubs.shape[1]

        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)

        text_global = text_outputs.pooler_output

        f_b = self.enc_b(pb)
        f_c = self.enc_c(pc)
        f_c_aligned = self.sa_ot(f_b, f_c)

        f_ubs_flat = self.enc_b(ubs.view(-1, 3, 256, 256))
        gap_ubs = F.adaptive_avg_pool2d(f_ubs_flat, (1, 1)).view(B, N, 768)
        att = F.softmax(self.mil_attn(gap_ubs).masked_fill(uv.unsqueeze(-1)==0, -100.0), dim=1)
        f_bag = torch.sum(att * gap_ubs, dim=1)

        gap_paired = F.adaptive_avg_pool2d(f_c_aligned, (1,1)).view(B, -1)
        combined_img_global = torch.cat([gap_paired, f_bag], dim=1) # [B, 768*2]

        img_align_feat = F.normalize(gap_paired, dim=-1)
        txt_align_feat = F.normalize(text_global, dim=-1)
        logit_scale = torch.clamp(self.logit_scale.exp(), max=100.0)

        logits_per_image = logit_scale * img_align_feat @ txt_align_feat.t()
        logits_per_text = logits_per_image.t()

        labels = torch.arange(B, dtype=torch.long, device=pb.device)
        loss_itc = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2

        concepts = self.concept_gen(combined_img_global)
        diag_node = self.global_img_gen(combined_img_global).unsqueeze(1)
        text_node = self.text_node_gen(text_global).unsqueeze(1)

        graph_nodes = torch.cat([torch.stack(concepts, dim=1), diag_node, text_node], dim=1)
        adj = F.softmax(self.adj_matrix.expand(B, -1, -1), dim=-1)
        gcn_out = self.gcn2(self.gcn1(graph_nodes, adj), adj)

        malignancy_pred = self.mal_head(gcn_out.view(B, -1))
        concept_preds = [self.concept_classifiers[i](concepts[i]) for i in range(self.num_concepts)]

        mask_pd_main = self.decoder_main(f_c_aligned, text_global)

        text_global_repeated = text_global.repeat_interleave(N, dim=0)
        mask_pd_bag = self.decoder_aux(f_ubs_flat, text_global_repeated).view(B, N, 1, 256, 256)

        ortho_loss = 0
        if self.training:
            c_tensor = F.normalize(torch.stack(concepts, dim=1), dim=-1)

            sim = torch.bmm(c_tensor, c_tensor.transpose(1, 2))
            identity = torch.eye(self.num_concepts, device=pc.device).unsqueeze(0)
            ortho_loss = torch.sum((sim - identity) ** 2) / B

        return mask_pd_main, mask_pd_bag, ortho_loss, loss_itc, concept_preds, malignancy_pred

if __name__ == "__main__":

    model = VL_CMGC_Net().cuda()
    pb = torch.randn(2, 3, 256, 256).cuda()
    pc = torch.randn(2, 3, 256, 256).cuda()
    ubs = torch.randn(2, 5, 3, 256, 256).cuda()
    uv = torch.ones(2, 5).cuda()

    input_ids = torch.randint(0, 21128, (2, 512)).cuda()
    attention_mask = torch.ones(2, 512).cuda()

    mask_main, mask_bag, o_loss, itc_loss, c_preds, mal_pred = model(pb, pc, ubs, uv, input_ids, attention_mask)
    print("Main Mask shape:", mask_main.shape)
    print("Malignancy Pred shape:", mal_pred.shape)
    print("InfoNCE Loss:", itc_loss.item())
    print("VL_CMGC_Net compiled successfully!")
