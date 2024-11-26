import torch.nn as nn
import torch
from transformers import ViTModel, ViTConfig

class SiameseViT(nn.Module):
    def __init__(self):
        super(SiameseViT, self).__init__()
        self.config = ViTConfig.from_pretrained('google/vit-base-patch16-224-in21k')
        self.encoder = ViTModel(self.config)
        self.fc = nn.Linear(self.config.hidden_size, 128)  # Project to 128-dim space

    def forward_one(self, x):
        x = self.encoder(x).last_hidden_state[:, 0, :]  # Use CLS token
        x = self.fc(x)
        return nn.functional.normalize(x, p=2, dim=1)  # L2 normalize

    def forward(self, anchor, positive, negative):
        a_enc = self.forward_one(anchor)
        p_enc = self.forward_one(positive)
        n_enc = self.forward_one(negative)
        return a_enc, p_enc, n_enc

# Triplet Loss
class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()
    


