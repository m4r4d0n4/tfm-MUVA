from datasets import load_dataset
import requests
from PIL import Image
from io import BytesIO
import os


'''
Class with the WikiArt (from HuggingFace) edited so we can get the item separed by the artist.

This class is modified so it can work for triplet loss
'''



from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random

class WikiArtTripletDataset(Dataset):
    def __init__(self, split="train"):
        self.ds = load_dataset("huggan/wikiart", split=split)
        len_dataset = len(self.ds)
        print(f"Imagenes totales {len_dataset}")
        self.artists = list(set(self.ds['artist']))
        self.artist_to_indices = {artist: [] for artist in self.artists}
        for i, item in enumerate(self.ds):
            print(f"Dataset number {i}")
            self.artist_to_indices[item['artist']].append(i)
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        anchor = self.ds[idx]
        anchor_artist = anchor['artist']
        
        # Get a positive sample (same artist)
        positive_idx = random.choice([i for i in self.artist_to_indices[anchor_artist] if i != idx])
        positive = self.ds[positive_idx]
        
        # Get a negative sample (different artist)
        negative_artist = random.choice([a for a in self.artists if a != anchor_artist])
        negative_idx = random.choice(self.artist_to_indices[negative_artist])
        negative = self.ds[negative_idx]
        
        return {
            'anchor': self.transform(anchor['image']),
            'positive': self.transform(positive['image']),
            'negative': self.transform(negative['image']),
            'artist': anchor_artist
        }
'''
# Create DataLoader
dataset = WikiArtTripletDataset()
#dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
for i in range(10):
    try:
        sample = dataset[i]
        print(f"Sample {i} loaded successfully")
        print(f"Anchor shape: {sample['anchor'].shape}")
        print(f"Positive shape: {sample['positive'].shape}")
        print(f"Negative shape: {sample['negative'].shape}")
    except Exception as e:
        print(f"Error at sample {i}: {e}")'''



























# Load the dataset in streaming mode
ds = load_dataset("huggan/wikiart", split="train", streaming=True)

# Take only one example
dst = ds.take(1)

# Create a directory to save the image if it doesn't exist
os.makedirs("wikiart_images", exist_ok=True)

# Iterate over the single item (we use next() to get the first item)
item = next(iter(dst))

# Print some information about the image
print(f"Artist: {item['artist']}")
print(f"Style: {item['style']}")

img = item['image']


# Save the image as PNG
filename = f"wikiart_images/test.png"
img.save(filename, format='PNG')

print(f"Image saved as: {filename}")