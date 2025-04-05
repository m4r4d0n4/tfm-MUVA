import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.cnn_dataset import WikiArtCNNDataset
from model.vit_scratch import ViTScratch
import matplotlib.pyplot as plt
import os

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the datasets
train_dataset = WikiArtCNNDataset(split="train")
val_dataset = WikiArtCNNDataset(split="validation")
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Load the model
num_classes = train_dataset.get_num_classes()
model = ViTScratch(num_classes).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
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
val_loss_values = []
val_accuracy_values = []

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_dataloader):
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
        for images, labels in val_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_dataloader)
    val_accuracy = 100 * correct / len(val_dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
    val_loss_values.append(val_loss)
    val_accuracy_values.append(val_accuracy)

print("Finished Training")

# Save the model
torch.save(model.state_dict(), 'vit_scratch_finetuned.pth')

# Plot the loss values
plt.plot(loss_values, label="Training Loss")
plt.plot(val_loss_values, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.savefig("graphs/train_vit_scratch_loss.png")
plt.close()

# Plot the validation accuracy values
plt.plot(val_accuracy_values)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy")
plt.savefig("graphs/train_vit_scratch_accuracy.png")
plt.close()
