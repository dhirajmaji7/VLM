import torch
import torch.nn as nn
import torch.nn.functional as F

from config import VitConfig

class VitEmbeddingLayer(nn.Module):
    def __init__(self, config: VitConfig):
        super().__init__()
        self.patch_embedding = nn.Conv2d(config.num_channels, 
                                         config.embd_dim, 
                                         kernel_size=config.patch_size, 
                                         stride=config.patch_size)
        self.num_patches = (config.img_size // config.patch_size) ** 2
        self.cls_embedding = nn.Embedding(1, config.embd_dim)
        self.position_embedding = nn.Embedding(self.num_patches + 1, config.embd_dim)

    def forward(self, images):
        # images: B, 3, H, W
        B, C, H, W = images.shape
        device = images.device
        embeddings = self.patch_embedding(images) # B, embd_dim, H/p, W/p
        embeddings = embeddings.flatten(start_dim=2).transpose(1, 2) # B, num_patches, embd_dim
        cls_token = self.cls_embedding(torch.tensor(0, device=device)) # 1, embd_dim
        cls_token = cls_token.unsqueeze(0).expand(B, -1, -1) # B, 1, embd_dim
        embeddings = torch.concat((cls_token, embeddings), dim=1) # B, num_patches, embd_dim
        position_ids = torch.arange(self.num_patches + 1, device=device)
        embeddings = embeddings + self.position_embedding(position_ids) # B, num_patches, embd_dim
        return embeddings # B, num_patches, embd_dim


class MultiHeadAttention(nn.Module):
    def __init__(self, config: VitConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.embd_dim // config.num_heads
        self.Q = nn.Linear(config.embd_dim, config.embd_dim)
        self.K = nn.Linear(config.embd_dim, config.embd_dim)
        self.V = nn.Linear(config.embd_dim, config.embd_dim)

    def forward(self, x):
        B, N, embd_dim = x.shape
        q = self.Q(x) # B, num_patches, embd_dim
        k = self.K(x)
        v = self.V(x)

        q = q.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # B, num_heads, num_patches, head_dim
        k = k.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn_weights = torch.matmul(q, k.transpose(2, 3)) / (self.head_dim**0.5) # B, nH, Np, Np
        attn_weights = F.softmax(attn_weights, dim=-1)
        out = torch.matmul(attn_weights, v) # B, nH, Np, head_dim

        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, -1) # B, Np, embd_dim
        return out


class MLP(nn.Module):
    def __init__(self, config: VitConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.embd_dim, config.embd_dim * 4)
        self.linear2 = nn.Linear(config.embd_dim * 4, config.embd_dim)

    def forward(self, x):
        x = F.gelu(self.linear1(x))
        x = self.linear2(x)
        return x


class VitEncoderBlock(nn.Module):
    def __init__(self, config: VitConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embd_dim)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.embd_dim)
        self.mlp = MLP(config)

    def forward(self, embeddings):
        x = self.ln1(embeddings)
        x = self.attn(x)
        residual = x + embeddings
        x = self.ln2(residual)
        x = self.mlp(x)
        x = x + residual
        return x

class VitEncoder(nn.Module):
    def __init__(self, config: VitConfig):
        super().__init__()
        self.encoder = nn.ModuleList([VitEncoderBlock(config) for n in range(config.num_layers)])

    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, config: VitConfig):
        super().__init__()
        self.embedding_layer = VitEmbeddingLayer(config)
        self.encoder = VitEncoder(config)
        self.mlp_head = nn.Linear(config.embd_dim, config.num_classes)
        
    def forward(self, images):
        # B, 3, H, W
        embeddings = self.embedding_layer(images) # B, N, C
        features = self.encoder(embeddings) # B, N, C
        cls_token = features[:, 0].squeeze() # B, C
        logits = self.mlp_head(cls_token) # B, num_classes
        return logits

