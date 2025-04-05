import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class SiameseResNet50(nn.Module):
    def __init__(self, embedding_dim=128):
        super(SiameseResNet50, self).__init__()
        # Load pre-trained ResNet50 model
        self.resnet50 = models.resnet50(pretrained=True)
        
        # Freeze all the layers
        for param in self.resnet50.parameters():
            param.requires_grad = False
        
        # Remove the classification head
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-1])
        
        self.embedding_dim = embedding_dim
        self.fc = nn.Linear(2048, embedding_dim)

    def forward_one(self, x):
        x = self.resnet50(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, img1, img2):
        feat1 = self.forward_one(img1)
        feat2 = self.forward_one(img2)
        return feat1, feat2

def contrastive_loss(embedding1, embedding2, label, margin=1.0):
    """
    Compute contrastive loss
    
    Args:
        embedding1 (torch.Tensor): Embedding of the first image
        embedding2 (torch.Tensor): Embedding of the second image
        label (torch.Tensor): 1 if the images are from the same class, 0 otherwise
        margin (float): Margin for the contrastive loss
    
    Returns:
        torch.Tensor: Contrastive loss
    """
    dist = F.pairwise_distance(embedding1, embedding2)
    loss = torch.mean((1-label) * torch.pow(dist, 2) +
                            (label) * torch.pow(torch.relu(margin - dist), 2))
    return loss

if __name__ == '__main__':
    # Example usage:
    model = SiameseResNet50()
    print(model)

    # Example input
    img1 = torch.randn(1, 3, 224, 224)
    img2 = torch.randn(1, 3, 224, 224)
    feat1, feat2 = model(img1, img2)
    print(feat1.shape)
    print(feat2.shape)

    label = torch.tensor([1], dtype=torch.float32)
    loss = contrastive_loss(feat1, feat2, label)
    print(loss)
