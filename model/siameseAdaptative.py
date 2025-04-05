import torch
import torch.nn as nn
import math
from torchvision.transforms.functional import resize
from transformers import ViTConfig, ViTModel

class Absolute2DPositionalEmbedding(nn.Module):
    def __init__(self, hidden_size=768):
        super().__init__()
        self.row_embed = nn.Embedding(1024, hidden_size)  # Máximo teórico
        self.col_embed = nn.Embedding(1024, hidden_size)
    
    def forward(self, grid_size):
        rows = torch.arange(grid_size[0], device=self.row_embed.weight.device)
        cols = torch.arange(grid_size[1], device=self.col_embed.weight.device)
        row_emb = self.row_embed(rows).unsqueeze(1)
        col_emb = self.col_embed(cols).unsqueeze(0)
        return (row_emb + col_emb).view(-1, row_emb.size(2))
    

class AspectAwareViT(nn.Module):
    def __init__(self, patch_size=16, max_seq_len=256, hidden_size=768):
        super().__init__()
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size
        
        # 1. Capa de extracción de parches personalizada
        self.patch_embed = nn.Conv2d(3, hidden_size, kernel_size=patch_size, stride=patch_size)
        
        # 2. Posicionales 2D
        self.pos_embed = Absolute2DPositionalEmbedding(hidden_size)
        
        # 3. Configurar solo el encoder del ViT
        self.config = ViTConfig(
            hidden_size=hidden_size,
            num_hidden_layers=12,
            num_attention_heads=12,
            position_embedding_type="none"  # Sin posicionales propias
        )
        self.transformer = ViTModel(self.config).encoder  # <--- Solo el encoder!
        
        # 4. Normalización y dropout como en ViT original
        self.layernorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

    def calculate_grid(self, H, W):
        aspect_ratio = W / H
        max_area = self.max_seq_len
        
        # Optimización para máxima cantidad de parches manteniendo relación de aspecto
        h = min(int((max_area * H**2 / (W * H))**0.5), H // self.patch_size)
        w = min(int(h * aspect_ratio), W // self.patch_size)
        
        return h, w

    def forward(self, x):
        B, C, H, W = x.shape
        h_grid, w_grid = self.calculate_grid(H, W)
        
        # Escalado inteligente
        new_H = h_grid * self.patch_size
        new_W = w_grid * self.patch_size
        x = resize(x, (new_H, new_W))
        
        # Extracción de parches (batch, hidden, h, w)
        patches = self.patch_embed(x)
        
        # Reorganizar a (batch, seq_len, hidden)
        patches = patches.flatten(2).transpose(1, 2)
        
        # Añadir posicionales 2D
        pos_emb = self.pos_embed((h_grid, w_grid)).unsqueeze(0).expand(B, -1, -1)
        patches += pos_emb
        
        # Capas de preprocesamiento ViT
        patches = self.dropout(patches)
        patches = self.layernorm(patches)
        
        # Pasar por el transformer
        sequence = self.transformer(patches).last_hidden_state
        
        return sequence.mean(dim=1)  # Pooling promedio
class SiameseArtNet(nn.Module):
    def __init__(self, max_seq_len=256):
        super().__init__()
        self.encoder = AspectAwareViT(max_seq_len=max_seq_len)
        self.projection = nn.Linear(768, 256)
    
    def forward_one(self, x):
        features = self.encoder(x)
        return nn.functional.normalize(self.projection(features), p=2, dim=1)
    
    def forward(self, anchor, positive, negative):
        return (self.forward_one(anchor),
                self.forward_one(positive),
                self.forward_one(negative))

class AdaptiveTripletLoss(nn.Module):
    def __init__(self, margin=0.2, alpha=0.1):
        super().__init__()
        self.margin = margin
        self.alpha = alpha
    
    def forward(self, anchor, positive, negative):
        pos_dist = torch.cdist(anchor, positive)
        neg_dist = torch.cdist(anchor, negative)
        
        dynamic_margin = self.margin + self.alpha * (pos_dist - neg_dist).detach()
        losses = torch.relu(pos_dist - neg_dist + dynamic_margin)
        return losses.mean()

    
model = SiameseArtNet(max_seq_len=512).cuda()
loss_fn = AdaptiveTripletLoss(margin=0.3, alpha=0.05)
# Inputs de diferentes tamaños en el mismo batch
anchor = torch.randn(4, 3, 1200, 900).cuda()  # Vertical
positive = torch.randn(4, 3, 800, 600).cuda()  # Horizontal
negative = torch.randn(4, 3, 1000, 600).cuda()  # Horizontal

# Forward pass
a_emb, p_emb, n_emb = model(anchor, positive, negative)
loss = loss_fn(a_emb, p_emb, n_emb)

print(a_emb)
print(p_emb)
print(n_emb)
print(loss)