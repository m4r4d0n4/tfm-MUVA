import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.dataset import WikiArtTripletDataset
from model.siamese_resnet import SiameseResNet50, contrastive_loss
import matplotlib.pyplot as plt
import os

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the datasets
train_dataset = WikiArtTripletDataset(split="train", siamese=True)
val_dataset = WikiArtTripletDataset(split="validation", siamese=True)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=10)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=10)

# Load the model
embedding_dim = 128
model = SiameseResNet50(embedding_dim).to(device)

# Define the loss function and optimizer
criterion = contrastive_loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create the graphs directory if it doesn't exist
if not os.path.exists("graphs"):
    os.makedirs("graphs")

# Training loop
num_epochs = 10

print("Training started")
start_time = time.time()

# Store the loss values for plotting
loss_values = []

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_dataloader):
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
        
        # Calculate estimated remaining time
        batches_done = (epoch * len(train_dataloader) + i)
        batches_left = num_epochs * len(train_dataloader) - batches_done
        time_per_batch = (time.time() - start_time) / (batches_done + 1)
        remaining_time = batches_left * time_per_batch
        
        if (i+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_dataloader)}], Loss: {running_loss/10:.4f}, Remaining time: {remaining_time/60:.2f} min")
            loss_values.append(running_loss / 10)
            running_loss = 0.0

    # Validation loop
    model.eval()
    val_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data in val_dataloader:
            anchor = data['anchor'].to(device)
            positive = data['positive'].to(device)
            label = data['label'].to(device)
            embedding1, embedding2 = model(anchor, positive)
            loss = criterion(embedding1, embedding2, label)
            val_loss += loss.item()
            
            # Calculate accuracy (assuming label 0 or 1)
            predictions = torch.sigmoid(embedding1.sub(embedding2).pow(2).sum(1))
            predicted_labels = (predictions < 0.5).float()
            correct += (predicted_labels == label).sum().item()

    val_loss /= len(val_dataloader)
    val_accuracy = 100 * correct / len(val_dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

print("Finished Training")

# Save the model
torch.save(model.state_dict(), 'siamese_resnet50_finetuned.pth')

# Plot the loss values
plt.plot(loss_values)
plt.xlabel("Step (x10)")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig("graphs/train_siamese_resnet_loss.png")
plt.close()
