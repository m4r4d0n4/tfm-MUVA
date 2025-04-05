import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from ..model.siamvit import SiameseViT, TripletLoss
from ..dataset.dataset import WikiArtTripletDataset
import matplotlib.pyplot as plt
import os


dataset = WikiArtTripletDataset()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseViT().to(device)
criterion = TripletLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 10

print("Training started")

# Store the loss values for plotting
loss_values = []

for epoch in range(num_epochs):
    print(f"Epoca numero {epoch}")
    model.train()
    total_loss = 0
    for i,batch in enumerate(dataloader):
        print(f"Batch numero {i}")     
        anchor = batch['anchor'].to(device)
        positive = batch['positive'].to(device)
        negative = batch['negative'].to(device)
        
        optimizer.zero_grad()
        a_enc, p_enc, n_enc = model(anchor, positive, negative)
        loss = criterion(a_enc, p_enc, n_enc)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}")
    loss_values.append(total_loss/len(dataloader))

# Create the graphs directory if it doesn't exist
if not os.path.exists("graphs"):
    os.makedirs("graphs")

# Save the model
torch.save(model.state_dict(), 'siamese_vit_wikiart.pth')

# Plot the loss values
plt.plot(loss_values)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig("graphs/train_siamese_vit_loss.png")
plt.close()
