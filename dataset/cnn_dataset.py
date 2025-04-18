from datasets import load_dataset
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class WikiArtCNNDataset(Dataset):
    def __init__(self, split="train", validation_split=0.2):
        # Load the entire dataset
        ds = load_dataset("huggan/wikiart", split="train")

        # Split into training and validation sets
        split_ds = ds.train_test_split(test_size=validation_split, seed=42)
        train_ds = split_ds["train"]
        val_ds = split_ds["test"]

        self.artists = list(set(train_ds["artist"]))
        self.artist_to_index = {artist: i for i, artist in enumerate(self.artists)}

        if split == "train":
            transform_list = [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=20),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
            self.transform = transforms.Compose(transform_list)
            self.ds = train_ds
        elif split == "validation":
            transform_list = [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
            self.transform = transforms.Compose(transform_list)
            self.ds = val_ds
        else:
            raise ValueError("Invalid split. Must be 'train' or 'validation'")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        image = self.transform(item['image'])
        artist_index = self.artist_to_index[item['artist']]
        return image, artist_index

    def get_num_classes(self):
        return len(self.artists)

    def get_artists(self):
        return self.artists

# Example usage:
if __name__ == '__main__':
    dataset = WikiArtCNNDataset(split="train")
    print(f"Number of images: {len(dataset)}")
    print(f"Number of classes: {dataset.get_num_classes()}")
    image, label = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Label: {label}")
