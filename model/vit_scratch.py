import torch
import torch.nn as nn
from transformers import ViTConfig, ViTModel

class ViTScratch(nn.Module):
    def __init__(self, num_classes):
        super(ViTScratch, self).__init__()
        # Load ViT configuration
        self.config = ViTConfig.from_pretrained('google/vit-base-patch16-224-in21k')
        # Modify the config to remove pretraining
        self.config.num_labels = num_classes
        self.config.hidden_dropout_prob = 0.0
        self.config.attention_dropout_prob = 0.0

        # Initialize ViT model from config
        self.vit = ViTModel(self.config)
        
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
    model = ViTScratch(num_classes)
    print(model)

    # Example input
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(output.shape)

    embedding = model.get_embedding(x)
    print(embedding.shape)
