import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from typing import List, Tuple

from inference import load_model, preprocess_image, compute_image_embedding

class FewShotEvaluator:
    def __init__(self, database_path: str, model_name: str, num_classes: int = 5, num_support: int = 3, num_query: int = 15):
        """
        Initializes the FewShotEvaluator.

        Args:
            database_path (str): Path to the embedding database (.npz file).
            model_name (str): Name of the model to use for embedding calculation ('siamese_vit', 'resnet50', 'siamese_resnet').
            num_classes (int): Number of classes to evaluate.
            num_support (int): Number of support images per class.
            num_query (int): Number of query images per class.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.database_path = database_path
        self.model_name = model_name
        self.num_classes = num_classes
        self.num_support = num_support
        self.num_query = num_query
        self.support_images = [[] for _ in range(num_classes)]  # List of lists to store support images for each class
        self.query_images = [[] for _ in range(num_classes)]  # List of lists to store query images for each class
        self.class_names = [f"artist_{i}" for i in range(num_classes)] # Assign class names

        # Load the model
        self.model = load_model(model_type=self.model_name).to(self.device)
        self.model.eval()

        # Load the embedding database
        try:
            database = np.load(self.database_path)
            self.embeddings = database['embeddings']
            self.artists = database['artists']
            self.image_ids = database['image_ids']
        except FileNotFoundError:
            raise FileNotFoundError(f"Embedding database not found at {self.database_path}")

        # Add support images to the database
        self.add_support_set_to_database()

    def add_support_image(self, class_index: int, image_path: str):
        """Adds a support image to the specified class."""
        if len(self.support_images[class_index]) < self.num_support:
            self.support_images[class_index].append(image_path)
        else:
            print(f"Support set for class {class_index} is full.")

    def add_query_image(self, class_index: int, image_path: str):
        """Adds a query image to the specified class."""
        if len(self.query_images[class_index]) < self.num_query:
            self.query_images[class_index].append(image_path)
        else:
            print(f"Query set for class {class_index} is full.")

    def add_support_set_to_database(self):
        """Adds the support set images to the embedding database."""
        for class_index in range(self.num_classes):
            for image_path in self.support_images[class_index]:
                image = preprocess_image(image_path).to(self.device)
                embedding = compute_image_embedding(self.model, image, self.model_name).cpu().numpy()

                # Add the embedding, artist (class name), and image ID to the database
                self.embeddings = np.vstack((self.embeddings, embedding))
                self.artists = np.append(self.artists, self.class_names[class_index])
                self.image_ids = np.append(self.image_ids, -1)  # Use -1 as a placeholder for support images

    def evaluate(self):
        """Evaluates the query images by extracting their embeddings."""
        query_embeddings = []
        for class_index in range(self.num_classes):
            for image_path in self.query_images[class_index]:
                image = preprocess_image(image_path).to(self.device)
                embedding = compute_image_embedding(self.model, image, self.model_name).cpu().numpy()
                query_embeddings.append(embedding)

        # For now, just print the number of query embeddings
        print(f"Number of query embeddings: {len(query_embeddings)}")

if __name__ == '__main__':
    # Example usage
    evaluator = FewShotEvaluator(database_path="wikiart_embeddings.npz", model_name="siamese_resnet")

    # Add support images (replace with actual image paths)
    for i in range(evaluator.num_classes):
        for j in range(evaluator.num_support):
            evaluator.add_support_image(i, f"support_image_class_{i}_{j}.jpg")

    # Add query images (replace with actual image paths)
    for i in range(evaluator.num_classes):
        for j in range(evaluator.num_query):
            evaluator.add_query_image(i, f"query_image_class_{i}_{j}.jpg")

    evaluator.evaluate()
