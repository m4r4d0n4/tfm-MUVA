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

        accuracy = self.calculate_accuracy(query_embeddings, self.embeddings, self.artists, self.class_names, self.num_classes, self.num_query)
        map_at_k = self.calculate_mean_average_precision(query_embeddings, self.embeddings, self.artists, self.class_names, self.num_classes, self.num_query)
        generalization_gap = self.calculate_generalization_gap(query_embeddings, self.embeddings, self.artists, self.class_names, self.num_classes, self.num_query)

        print(f"Few-Shot Accuracy: {accuracy}")
        print(f"Mean Average Precision (mAP@K): {map_at_k}")
        print(f"Few-Shot Generalization Gap: {generalization_gap}")

    def calculate_accuracy(self, query_embeddings: List[np.ndarray], embeddings: np.ndarray, artists: np.ndarray, class_names: List[str], num_classes: int, num_query: int, k: int = 1) -> float:
        """Calculates the few-shot accuracy (N-shot-K-way)."""
        correct = 0
        total = 0
        for class_index in range(num_classes):
            for i in range(num_query):
                query_embedding = query_embeddings[class_index * num_query + i]
                # Calculate distances to all embeddings in the database
                distances = np.linalg.norm(embeddings - query_embedding, axis=1)
                # Get the indices of the k-nearest neighbors
                knn_indices = np.argsort(distances)[:k]
                # Get the predicted class names of the k-nearest neighbors
                knn_labels = artists[knn_indices]
                # Determine the most frequent class among the k-nearest neighbors
                predicted_class = np.argmax(np.bincount(np.array([class_names.index(label) for label in knn_labels])))
                # Check if the prediction is correct
                if predicted_class == class_index:
                    correct += 1
                total += 1
        return correct / total

    def calculate_mean_average_precision(self, query_embeddings: List[np.ndarray], embeddings: np.ndarray, artists: np.ndarray, class_names: List[str], num_classes: int, num_query: int, k: int = 10) -> float:
        """Calculates the Mean Average Precision (mAP@K)."""
        map_sum = 0
        for class_index in range(num_classes):
            ap_sum = 0
            for i in range(num_query):
                query_embedding = query_embeddings[class_index * num_query + i]
                # Calculate distances to all embeddings in the database
                distances = np.linalg.norm(embeddings - query_embedding, axis=1)
                # Get the indices of the k-nearest neighbors
                knn_indices = np.argsort(distances)[:k]
                # Get the predicted class names of the k-nearest neighbors
                knn_labels = artists[knn_indices]
                # Calculate precision at each rank
                precision_sum = 0
                num_correct = 0
                for j in range(k):
                    if class_names.index(artists[class_index * num_query + i]) == np.argmax(np.bincount(np.array([class_names.index(label) for label in knn_labels[:j+1]]))):
                        num_correct += 1
                    precision_sum += num_correct / (j + 1)
                ap_sum += precision_sum / k
            map_sum += ap_sum / num_query
        return map_sum / num_classes

    def calculate_generalization_gap(self, query_embeddings: List[np.ndarray], embeddings: np.ndarray, artists: np.ndarray, class_names: List[str], num_classes: int, num_query: int, k: int = 1) -> float:
        """Calculates the Few-Shot Generalization Gap."""
        accuracy = self.calculate_accuracy(query_embeddings, embeddings, artists, class_names, num_classes, num_query, k)
        # For now, assume a fixed baseline accuracy (replace with actual baseline if available)
        baseline_accuracy = 0.5
        return accuracy - baseline_accuracy

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
