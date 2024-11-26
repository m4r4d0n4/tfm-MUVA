import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from model.siamvit import SiameseViT, TripletLoss
from dataset.dataset import WikiArtTripletDataset


dataset = WikiArtTripletDataset()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseViT().to(device)
criterion = TripletLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in dataloader:
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

# Save the model
torch.save(model.state_dict(), 'siamese_vit_wikiart.pth')