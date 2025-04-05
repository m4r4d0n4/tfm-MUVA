import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig

class ViTFineTune(nn.Module):
    def __init__(self, num_classes):
        super(ViTFineTune, self).__init__()
        # Load pre-trained ViT model
        self.config = ViTConfig.from_pretrained('google/vit-base-patch16-224-in21k')
        self.vit = ViTModel(self.config)
        
        # Freeze all the layers
        for param in self.vit.parameters():
            param.requires_grad = False
        
        # Replace the classification head
        self.num_features = self.vit.config.hidden_size
        self.classifier = nn.Linear(self.num_features, num_classes)

    def forward(self, x):
        x = self.vit(x).last_hidden_state[:, 0, :]  # Use CLS token
        x = self.classifier(x)
        return x
    
    def get_embedding(self, x):
        # Extract features before the classification layer
        x = self.vit(x).last_hidden_state[:, 0, :]  # Use CLS token
        return x

if __name__ == '__main__':
    # Example usage:
    num_classes = 100 # Replace with the actual number of classes
    model = ViTFineTune(num_classes)
    print(model)

    # Example input
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(output.shape)

    embedding = model.get_embedding(x)
    print(embedding.shape)
