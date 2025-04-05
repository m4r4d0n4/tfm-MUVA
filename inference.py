import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from model.siamvit import SiameseViT
from model.cnn_finetune import ResNet50FineTune
from model.siamese_resnet import SiameseResNet50

def load_model(model_type='siamese_vit', model_path='siamese_vit_wikiart.pth'):
    """
    Load the pre-trained model
    
    Args:
        model_type (str): Type of model to use ('siamese_vit', 'resnet50', 'siamese_resnet')
        model_path (str): Path to the saved model weights
    
    Returns:
        Loaded model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_type == 'siamese_vit':
        model = SiameseViT().to(device)
    elif model_type == 'resnet50':
        model = ResNet50FineTune().to(device)
    elif model_type == 'siamese_resnet':
        model = SiameseResNet50().to(device)
    else:
        raise ValueError(f"Invalid model_type: {model_type}")
        
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

def compute_image_embedding(model, image_path, model_type='siamese_vit'):
    """
    Compute embedding for a single image
    
    Args:
        model: Loaded model
        image_path (str): Path to the image file
    
    Returns:
        Image embedding vector
    """
    device = next(model.parameters()).device
    image_tensor = preprocess_image(image_path).to(device)
    
    with torch.no_grad():
        if model_type == 'siamese_vit':
            embedding = model.encode_image(image_tensor)
        else:
            embedding = model.get_embedding(image_tensor)
    
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

def load_embedding_database(embedding_db_path):
    """
    Load the embedding database from a file.
    
    Args:
        embedding_db_path (str): Path to the embedding database file.
    
    Returns:
        tuple: A tuple containing the embeddings, artist names, and image IDs.
    """
    data = np.load(embedding_db_path)
    embeddings = data['embeddings']
    artists = data['artists']
    image_ids = data['image_ids']
    return embeddings, artists, image_ids

def search_embedding_database(query_embedding, embeddings, top_k=5):
    """
    Search the embedding database for the most similar images to a query embedding.
    
    Args:
        query_embedding (numpy.ndarray): The query embedding.
        embeddings (numpy.ndarray): The embeddings in the database.
        top_k (int): The number of most similar images to return.
    
    Returns:
        list: A list of tuples containing the artist name and image ID of the most similar images.
    """
    similarities = cosine_similarity(query_embedding.reshape(1, -1), embeddings)[0]
    
    # Get the indices of the top k most similar images
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    return top_indices

def main():
    """
    Example usage of the Siamese ViT model for inference
    """
    # Load the pre-trained model
    model_type = 'siamese_vit'
    model = load_model(model_type=model_type, model_path='siamese_vit_wikiart.pth')
    
    # Example: Compute embeddings for two images
    image1_path = './wikiart_images/test.png'
    
    # Load embedding database
    embedding_db_path = 'siamese_vit_embeddings.npz'
    embeddings, artists, image_ids = load_embedding_database(embedding_db_path)
    
    # Compute embedding for the query image
    query_embedding = compute_image_embedding(model, image1_path, model_type)
    
    # Search the embedding database
    top_indices = search_embedding_database(query_embedding, embeddings)
    
    print("Similar Artists:")
    for idx in top_indices:
        print(f"Artist: {artists[idx]}, Image ID: {image_ids[idx]}")

if __name__ == "__main__":
    main()
