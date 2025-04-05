import torch
import faiss
import numpy as np
import os
from tqdm import tqdm
from typing import List, Tuple

from model.siamvit import SiameseViT
from dataset.dataset import WikiArtTripletDataset

class EmbeddingDatabase:
    def __init__(self, model_path='siamese_vit_wikiart.pth', embedding_dim=768):
        """
        Initialize embedding database with Faiss index
        
        Args:
            model_path (str): Path to pre-trained model
            embedding_dim (int): Dimensionality of embedding vectors
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pre-trained model
        self.model = SiameseViT().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Create Faiss index (using GPU if available)
        self.index = faiss.IndexFlatL2(embedding_dim)  # L2 distance
        
        # Store metadata
        self.embeddings = []
        self.labels = []
        
    def generate_embeddings(self, image_folder: str):
        """
        Generate embeddings for all images in a folder
        
        Args:
            image_folder (str): Path to folder containing images
        """
        for filename in tqdm(os.listdir(image_folder)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                full_path = os.path.join(image_folder, filename)
                
                # Extract artist name from filename or folder structure
                artist = filename.split('_')[0]  # Assumes filename format like 'artistname_painting.jpg'
                
                try:
                    embedding = self._compute_embedding(full_path)
                    self.embeddings.append(embedding)
                    self.labels.append(artist)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
        
        # Convert to numpy array for Faiss
        embeddings_array = np.array(self.embeddings).astype('float32')
        self.index.add(embeddings_array)
        
    def _compute_embedding(self, image_path: str) -> np.ndarray:
        """
        Compute embedding for a single image
        
        Args:
            image_path (str): Path to image file
        
        Returns:
            Numpy array of embedding
        """
        image_tensor = preprocess_image(image_path).to(self.device)
        
        with torch.no_grad():
            embedding = self.model.encode_image(image_tensor)
        
        return embedding.cpu().numpy().flatten()
    
    def search_similar_images(self, query_image_path: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find most similar images to a query image
        
        Args:
            query_image_path (str): Path to query image
            top_k (int): Number of top similar images to return
        
        Returns:
            List of tuples with (artist, distance)
        """
        query_embedding = self._compute_embedding(query_image_path)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Perform similarity search
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Retrieve and return results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            results.append((self.labels[idx], dist))
        
        return results
    
    def save_index(self, save_path: str):
        """
        Save Faiss index and metadata
        
        Args:
            save_path (str): Path to save index and metadata
        """
        faiss.write_index(self.index, f"{save_path}_index.faiss")
        np.save(f"{save_path}_labels.npy", np.array(self.labels))
    
    def load_index(self, load_path: str):
        """
        Load previously saved Faiss index and metadata
        
        Args:
            load_path (str): Path to load index and metadata
        """
        self.index = faiss.read_index(f"{load_path}_index.faiss")
        self.labels = list(np.load(f"{load_path}_labels.npy"))

def preprocess_image(image_path, img_size=224):
    """
    Preprocess an image for the Siamese ViT model
    
    Args:
        image_path (str): Path to the image file
        img_size (int): Target image size for resizing
    
    Returns:
        Preprocessed image tensor
    """
    from PIL import Image
    import torchvision.transforms as transforms
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # Standard ImageNet normalization
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

def main():
    # Create or load embedding database
    db = EmbeddingDatabase()
    
    # Generate embeddings from a large image folder
    db.generate_embeddings('./wikiart_images')
    
    # Optional: Save the index for future use
    #db.save_index('wikiart_embeddings')
    
    # Search similar images
    query_image = './wikiart_images/test.jpg'
    similar_images = db.search_similar_images(query_image, top_k=5)
    
    print("Similar Artists:")
    for artist, distance in similar_images:
        print(f"Artist: {artist}, Distance: {distance}")

if __name__ == "__main__":
    main()