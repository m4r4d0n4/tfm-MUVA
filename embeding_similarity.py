import torch
import numpy as np
import os
import argparse
from tqdm import tqdm
from typing import List, Tuple
from PIL import Image
import torchvision.transforms as transforms

from model.siamvit import SiameseViT
from model.cnn_finetune import ResNet50FineTune
from model.siamese_resnet import SiameseResNet50
from inference import load_model, preprocess_image, compute_image_embedding

def find_closest_embedding(image_path: str, model_name: str, database_path: str, search_method: str = 'cosine') -> Tuple[str, float]:
    """
    Finds the closest embedding in a database to a given image.

    Args:
        image_path (str): Path to the input image.
        model_name (str): Name of the model to use for embedding calculation ('siamese_vit', 'resnet50', 'siamese_resnet').
        database_path (str): Path to the embedding database (.npz file).
        search_method (str): Method to use for searching the closest embedding ('cosine', 'euclidean').

    Returns:
        Tuple[str, float]: Tuple containing the artist name of the closest embedding and the distance.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = load_model(model_type=model_name).to(device)
    model.eval()

    # Load the embedding database
    try:
        database = np.load(database_path)
        embeddings = database['embeddings']
        artists = database['artists']
    except FileNotFoundError:
        raise FileNotFoundError(f"Embedding database not found at {database_path}")

    # Preprocess the image and compute its embedding
    image = preprocess_image(image_path).to(device)
    embedding = compute_image_embedding(model, image, model_name).cpu().numpy()

    def cosine_similarity(embedding, embeddings):
        embedding = embedding / np.linalg.norm(embedding)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        distances = np.dot(embeddings, embedding.T).flatten()
        closest_index = np.argmax(distances)
        distance = distances[closest_index]  # Similarity score
        return closest_index, distance

    def euclidean_distance(embedding, embeddings):
        distances = np.linalg.norm(embeddings - embedding, axis=1)
        closest_index = np.argmin(distances)
        distance = distances[closest_index]
        return closest_index, distance

    # Search for the closest embedding
    if search_method == 'cosine':
        closest_index, distance = cosine_similarity(embedding, embeddings)
    elif search_method == 'euclidean':
        closest_index, distance = euclidean_distance(embedding, embeddings)
    else:
        raise ValueError(f"Invalid search method: {search_method}")

    closest_artist = artists[closest_index]

    return closest_artist, distance


def main():
    parser = argparse.ArgumentParser(description="Find the closest embedding in a database to a given image.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use ('siamese_vit', 'resnet50', 'siamese_resnet').")
    parser.add_argument("--database_path", type=str, required=True, help="Path to the embedding database (.npz file).")
    parser.add_argument("--search_method", type=str, default="cosine", help="Method to use for searching the closest embedding ('cosine', 'euclidean').")

    args = parser.parse_args()

    try:
        closest_artist, distance = find_closest_embedding(args.image_path, args.model_name, args.database_path, args.search_method)
        print(f"Closest artist: {closest_artist}, Distance: {distance}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
