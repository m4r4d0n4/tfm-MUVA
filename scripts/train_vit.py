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

# Load the datasets
train_dataset = WikiArtCNNDataset(split="train")
val_dataset = WikiArtCNNDataset(split="validation")
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=10)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=10)

# Load the model
num_classes = train_dataset.get_num_classes()
model = ViTFineTune(num_classes).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create the graphs directory if it doesn't exist
if not os.path.exists("graphs"):
    os.makedirs("graphs")

# Training loop
num_epochs = 300

print("Training started")
start_time = time.time()

# Store the loss values for plotting
loss_values = []

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

    # Save the model every 50 epochs
    if (epoch + 1) % 50 == 0:
        model_path = f"vit_finetuned_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

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
