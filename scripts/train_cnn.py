import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from dataset.cnn_dataset import WikiArtCNNDataset
from model.cnn_finetune import ResNet50FineTune
import matplotlib.pyplot as plt
import os

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset
dataset = WikiArtCNNDataset(split="train")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# Load the model
num_classes = dataset.get_num_classes()
model = ResNet50FineTune(num_classes).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create the graphs directory if it doesn't exist
if not os.path.exists("graphs"):
    os.makedirs("graphs")

# Training loop
num_epochs = 10

print("Training started")

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
        
        if (i+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {running_loss/10:.4f}")
            loss_values.append(running_loss / 10)
            running_loss = 0.0

print("Finished Training")

# Save the model
torch.save(model.state_dict(), 'resnet50_finetuned.pth')

# Plot the loss values
plt.plot(loss_values)
plt.xlabel("Step (x10)")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig("graphs/train_cnn_loss.png")
plt.close()
