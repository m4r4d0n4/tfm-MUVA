import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.dataset import WikiArtTripletDataset
from model.siamese_resnet import SiameseResNet50, contrastive_loss

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset
dataset = WikiArtTripletDataset(split="train", siamese=True)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# Load the model
embedding_dim = 128
model = SiameseResNet50(embedding_dim).to(device)

# Define the loss function and optimizer
criterion = contrastive_loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10

print("Training started")

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    model.train()
    running_loss = 0.0
    for i, data in enumerate(dataloader):
        anchor = data['anchor'].to(device)
        positive = data['positive'].to(device)
        label = data['label'].to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        embedding1, embedding2 = model(anchor, positive)
        loss = criterion(embedding1, embedding2, label)
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if (i+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {running_loss/10:.4f}")
            running_loss = 0.0

print("Finished Training")

# Save the model
torch.save(model.state_dict(), 'siamese_resnet50_finetuned.pth')
