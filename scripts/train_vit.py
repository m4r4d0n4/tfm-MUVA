import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.cnn_dataset import WikiArtCNNDataset
from model.vit_finetune import ViTFineTune
import matplotlib.pyplot as plt
import os

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset
dataset = WikiArtCNNDataset(split="train")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# Load the model
num_classes = dataset.get_num_classes()
model = ViTFineTune(num_classes).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create the graphs directory if it doesn't exist
if not os.path.exists("graphs"):
    os.makedirs("graphs")

# Training loop
num_epochs = 1

print("Training started")
start_time = time.time()

# Store the loss values for plotting
loss_values = []

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calculate estimated remaining time
        batches_done = (epoch * len(dataloader) + i)
        batches_left = num_epochs * len(dataloader) - batches_done
        time_per_batch = (time.time() - start_time) / (batches_done + 1)
        remaining_time = batches_left * time_per_batch
        
        if (i+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {running_loss/10:.4f}, Remaining time: {remaining_time/60:.2f} min")
            loss_values.append(running_loss / 10)
            running_loss = 0.0

print("Finished Training")

# Save the model
torch.save(model.state_dict(), 'vit_finetuned.pth')

# Plot the loss values
plt.plot(loss_values)
plt.xlabel("Step (x10)")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig("graphs/train_vit_loss.png")
plt.close()
