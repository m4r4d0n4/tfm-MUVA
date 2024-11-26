import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        return x

class DynamicPositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        self.embed_dim = embed_dim
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :]

class FlexibleViT(nn.Module):
    def __init__(self, patch_size, embed_dim, num_heads, num_layers):
        super().__init__()
        self.patch_embed = PatchEmbedding(patch_size, embed_dim)
        self.pos_encoding = DynamicPositionalEncoding(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc(x)
        return x

class SiameseViT(nn.Module):
    def __init__(self, patch_size=16, embed_dim=768, num_heads=12, num_layers=12):
        super().__init__()
        self.vit = FlexibleViT(patch_size, embed_dim, num_heads, num_layers)

    def forward(self, img1, img2):
        feat1 = self.vit(img1)
        feat2 = self.vit(img2)
        distance = F.pairwise_distance(feat1, feat2)
        similarity = F.cosine_similarity(feat1, feat2)
        
        return feat1, feat2,distance, similarity

# Usage
model = SiameseViT()
img1 = torch.randn(1, 3, 224, 224)  # Can be any size
img2 = torch.randn(1, 3, 256, 320)  # Different size
feat1, feat2,d, sim = model(img1, img1)

print(d)