import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import random

def load_model(model_name, model_paths, device, num_classes=None):
    """
    Loads a trained model from the specified path.
    """
    if model_name not in model_paths:
        raise ValueError(f"Model path not found for model name: {model_name}")

    model_path = model_paths[model_name]

    if model_name == "cnn_finetune":
        from model.cnn_finetune import ResNet50FineTune
        model = ResNet50FineTune(num_classes=num_classes)
    elif model_name == "siameseAdaptative":
        from model.siameseAdaptative import SiameseArtNet
        model = SiameseArtNet()
    elif model_name == "siamese_resnet":
        from model.siamese_resnet import SiameseResNet50
        model = SiameseResNet50()
    elif model_name == "siamvit":
        from model.siamvit import SiameseViT
        model = SiameseViT()
    elif model_name == "vit_finetune":
        from model.vit_finetune import ViTFineTune
        model = ViTFineTune(num_classes=num_classes)
    elif model_name == "vit_scratch":
        from model.vit_scratch import ViTScratch
        model = ViTScratch(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set the model to evaluation mode

    return model

def load_dataset(author_dir, model_name, image_size=(224, 224)):
    """
    Loads the dataset from the specified author directory.
    """
    support_dir = os.path.join(author_dir, "support")
    query_dir = os.path.join(author_dir, "query")

    support_images = []
    query_images = []

    if model_name == "siameseAdaptative":
        transform = transforms.Compose([])
    else:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    for filename in os.listdir(support_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(support_dir, filename)
            img = Image.open(img_path).convert("RGB")
            img = transform(img)
            support_images.append(img)

    for filename in os.listdir(query_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(query_dir, filename)
            img = Image.open(img_path).convert("RGB")
            img = transform(img)
            query_images.append(img)

    return torch.stack(support_images), torch.stack(query_images)

def extract_embeddings(model, images, device):
    """
    Extracts embeddings from the images using the loaded model.
    """
    model.eval()
    model.to(device)
    with torch.no_grad():
        embeddings = []
        for image in images:
            image = image.unsqueeze(0)  # Add batch dimension
            embedding = model.forward_one(image.to(device))
            embeddings.append(embedding.cpu().numpy())
        embeddings = np.concatenate(embeddings, axis=0)  # Concatenate embeddings
    return embeddings

def apply_kde(support_embeddings, query_embedding):
    """
    Applies the KDE algorithm to calculate the log-probability of the query embedding
    belonging to the same class as the support embeddings.
    """
    kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(support_embeddings)
    log_prob = kde.score_samples(query_embedding.reshape(1, -1))[0]
    return log_prob

def evaluate(model_name, dataset_dir, model_paths, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Evaluates the model on the dataset using KDE.
    """
    # Calculate the number of authors
    num_authors = len([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d)) and d.startswith("author_")])
    num_classes = num_authors  # Set num_classes to the number of authors
    device = torch.device(device)
    model = load_model(model_name, model_paths, device, num_classes=num_classes)

    all_labels = []
    all_log_probs = []

    for i in range(1, num_authors + 1):
        positive_author_dir = os.path.join(dataset_dir, f"author_{i}")
        negative_author_index = i + 8
        if negative_author_index > num_authors:
            negative_author_index -= num_authors
        negative_author_dir = os.path.join(dataset_dir, f"author_{negative_author_index}")

        support_images, _ = load_dataset(positive_author_dir, model_name)
        _, positive_query_images = load_dataset(positive_author_dir, model_name)
        _, negative_query_images = load_dataset(negative_author_dir, model_name)

        support_embeddings = extract_embeddings(model, support_images, device)
        positive_query_embeddings = extract_embeddings(model, positive_query_images, device)
        negative_query_embeddings = extract_embeddings(model, negative_query_images, device)

        # Apply KDE to each query image
        for query_embedding in positive_query_embeddings:
            log_prob = apply_kde(support_embeddings, query_embedding)
            all_log_probs.append(log_prob)
            all_labels.append(1)  # Positive label

        for query_embedding in negative_query_embeddings:
            log_prob = apply_kde(support_embeddings, query_embedding)
            all_log_probs.append(log_prob)
            all_labels.append(0)  # Negative label

        return all_labels, all_log_probs


def plot_roc_curve(labels, log_probs, model_name):
    """
    Plots the ROC curve.
    """
    fpr, tpr, thresholds = roc_curve(labels, log_probs)
    plt.plot(fpr, tpr, label=f'{model_name} ROC')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(f"roc_curve_{model_name}.png")
    plt.show()

if __name__ == "__main__":
    model_name = "cnn_finetune"  # Change this to the model you want to evaluate
    dataset_dir = "eval_dataset"
    model_paths = {  # Replace with the actual paths to the trained models
        "cnn_finetune": "cnn_finetune.pth",
        "siameseAdaptative": "siameseAdaptative.pth",
        "siamese_resnet": "siamese_resnet.pth",
        "siamvit": "siamvit.pth",
        "vit_finetune": "vit_finetune.pth",
        "vit_scratch": "vit_scratch.pth",
    }
    labels, log_probs = evaluate(model_name, dataset_dir, model_paths)
    plot_roc_curve(labels, log_probs, model_name)
