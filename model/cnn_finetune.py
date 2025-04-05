import torch
import torch.nn as nn
from torchvision import models

class ResNet50FineTune(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50FineTune, self).__init__()
        # Load pre-trained ResNet50 model
        self.resnet50 = models.resnet50(pretrained=True)
        
        # Freeze all the layers
        for param in self.resnet50.parameters():
            param.requires_grad = False
        
        # Replace the classification head
        num_ftrs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.resnet50(x)
        return x

    def get_embedding(self, x):
        # Extract features before the classification layer
        modules = list(self.resnet50.children())[:-1]
        resnet = nn.Sequential(*modules)
        embedding = resnet(x)
        embedding = torch.flatten(embedding, 1)
        return embedding

if __name__ == '__main__':
    # Example usage:
    num_classes = 100 # Replace with the actual number of classes
    model = ResNet50FineTune(num_classes)
    print(model)

    # Example input
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(output.shape)

    embedding = model.get_embedding(x)
    print(embedding.shape)
