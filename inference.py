import torch
import torchvision.transforms as transforms
from PIL import Image

from model.siamvit import SiameseViT

def load_model(model_path='siamese_vit_wikiart.pth'):
    """
    Load the pre-trained Siamese ViT model
    
    Args:
        model_path (str): Path to the saved model weights
    
    Returns:
        SiameseViT model loaded with pre-trained weights
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseViT().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set the model to evaluation mode
    return model

def preprocess_image(image_path, img_size=224):
    """
    Preprocess an image for the Siamese ViT model
    
    Args:
        image_path (str): Path to the image file
        img_size (int): Target image size for resizing
    
    Returns:
        Preprocessed image tensor
    """
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

def compute_image_embedding(model, image_path):
    """
    Compute embedding for a single image
    
    Args:
        model (SiameseViT): Loaded Siamese ViT model
        image_path (str): Path to the image file
    
    Returns:
        Image embedding vector
    """
    device = next(model.parameters()).device
    image_tensor = preprocess_image(image_path).to(device)
    
    with torch.no_grad():
        embedding = model.encode_image(image_tensor)
    
    return embedding.cpu().numpy()

def compute_similarity(embedding1, embedding2):
    """
    Compute cosine similarity between two image embeddings
    
    Args:
        embedding1 (numpy.ndarray): First image embedding
        embedding2 (numpy.ndarray): Second image embedding
    
    Returns:
        Cosine similarity score between the two embeddings
    """
    embedding1_norm = embedding1 / np.linalg.norm(embedding1)
    embedding2_norm = embedding2 / np.linalg.norm(embedding2)
    return np.dot(embedding1_norm, embedding2_norm)

def main():
    """
    Example usage of the Siamese ViT model for inference
    """
    # Load the pre-trained model
    model = load_model()
    
    # Example: Compute embeddings for two images
    image1_path = 'path/to/first/image.jpg'
    image2_path = 'path/to/second/image.jpg'
    
    embedding1 = compute_image_embedding(model, image1_path)
    embedding2 = compute_image_embedding(model, image2_path)
    
    # Compute similarity
    similarity = compute_similarity(embedding1, embedding2)
    print(f"Similarity between images: {similarity:.4f}")

if __name__ == "__main__":
    import numpy as np
    main()