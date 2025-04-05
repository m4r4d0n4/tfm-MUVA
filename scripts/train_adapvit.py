import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from model.siameseAdaptative import SiameseArtNet, AdaptiveTripletLoss
from dataset.datasetNoresize import WikiArtTripletDatasetNoResize
import matplotlib.pyplot as plt
import os
import torch.nn as nn

def custom_collate_fn(batch):
    anchors = [item['anchor'] for item in batch]
    positives = [item['positive'] for item in batch]
    negatives = [item['negative'] for item in batch]
    artists = [item['artist'] for item in batch]

    return {
        'anchor': anchors,  # Lista de tensores con tamaños variables
        'positive': positives,
        'negative': negatives,
        'artist': artists
    }

train_dataset =  WikiArtTripletDatasetNoResize(split="train")
val_dataset = WikiArtTripletDatasetNoResize(split="validation")
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4,collate_fn=custom_collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4,collate_fn=custom_collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseArtNet().to(device)
criterion = AdaptiveTripletLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 2

print("Training started")

# Store the loss values for plotting
loss_values = []
start_time = time.time()

for epoch in range(num_epochs):
    epoch_start = time.time()  # Tiempo de inicio de la época
    print(f"Época número {epoch+1}/{num_epochs}")
    
    model.train()
    total_loss = 0

    for i, batch in enumerate(train_dataloader):
        print(f"  → Procesando batch número {i+1}/{len(train_dataloader)}")
        
        anchors = batch['anchor']
        positives = batch['positive']
        negatives = batch['negative']
        
        anchors = torch.stack([img.to(device) for img in anchors])
        positives = torch.stack([img.to(device) for img in positives])
        negatives = torch.stack([img.to(device) for img in negatives])
        
        optimizer.zero_grad()
        a_enc, p_enc, n_enc = model(anchors, positives, negatives)
        loss = criterion(a_enc, p_enc, n_enc)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

        # Calculate estimated remaining time
        batches_done = (epoch * len(train_dataloader) + i)
        batches_left = num_epochs * len(train_dataloader) - batches_done
        time_per_batch = (time.time() - start_time) / (batches_done + 1)
        remaining_time = batches_left * time_per_batch
    
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_dataloader):.4f}, Remaining time: {remaining_time/60:.2f} min")
        loss_values.append(total_loss/len(train_dataloader))

    # Validation loop
    model.eval()
    val_loss = 0.0
    correct = 0
    with torch.no_grad():
        for batch in val_dataloader:
            anchors = batch['anchor']
            positives = batch['positive']
            negatives = batch['negative']
            
            anchors = torch.stack([img.to(device) for img in anchors])
            positives = torch.stack([img.to(device) for img in positives])
            negatives = torch.stack([img.to(device) for img in negatives])
            
            a_enc, p_enc, n_enc = model(anchors, positives, negatives)
            loss = criterion(a_enc, p_enc, n_enc)
            val_loss += loss.item()
            
            # Calculate triplet accuracy (approximate)
            triplet_correct = (loss < 1.0).sum().item()
            correct += triplet_correct

    val_loss /= len(val_dataloader)
    val_accuracy = 100 * correct / len(val_dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

# Create the graphs directory if it doesn't exist
if not os.path.exists("graphs"):
    os.makedirs("graphs")

# Save the model
torch.save(model.state_dict(), 'siameseadaptative_vit_wikiart.pth')

# Plot the loss values
plt.plot(loss_values)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig("graphs/train_siamese_adaptative_loss.png")
plt.close()
