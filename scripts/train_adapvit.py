import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch.optim as optim
import torch
import time
from torch.utils.data import DataLoader
from ..model.siameseAdaptative import SiameseArtNet, AdaptiveTripletLoss
from ..dataset.datasetNoresize import WikiArtTripletDatasetNoResize
import matplotlib.pyplot as plt
import os

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

dataset = WikiArtTripletDatasetNoResize()
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4,collate_fn=custom_collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseArtNet().to(device)
criterion = AdaptiveTripletLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 2

print("Training started")
start_time = time.time()  # Guarda el tiempo de inicio

# Store the loss values for plotting
loss_values = []

for epoch in range(num_epochs):
    epoch_start = time.time()  # Tiempo de inicio de la época
    print(f"Época número {epoch+1}/{num_epochs}")
    
    model.train()
    total_loss = 0

    for i, batch in enumerate(dataloader):
        print(f"  → Procesando batch número {i+1}/{len(dataloader)}")
        
        anchors = torch.stack([img.to(device) for img in batch['anchor']]).to(device)
        positives = torch.stack([img.to(device) for img in batch['positive']]).to(device)
        negatives = torch.stack([img.to(device) for img in batch['negative']]).to(device)
        
        optimizer.zero_grad()
        a_enc, p_enc, n_enc = model(anchors, positives, negatives)
        loss = criterion(a_enc, p_enc, n_enc)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    # Calcula tiempos
    epoch_duration = time.time() - epoch_start
    elapsed_time = time.time() - start_time
    estimated_total_time = (elapsed_time / (epoch + 1)) * num_epochs
    remaining_time = estimated_total_time - elapsed_time
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}")
    print(f"Tiempo de esta época: {epoch_duration:.2f}s")
    print(f"Tiempo estimado restante: {remaining_time/60:.2f} min\n")
    loss_values.append(total_loss/len(dataloader))

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
