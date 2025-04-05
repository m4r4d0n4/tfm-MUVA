# tfm-MUVA

## Overview

This project explores different deep learning architectures for artist attribution in paintings. It aims to develop an automatic similarity system that can associate a given artwork with the correct artist, even for artists the model has not been explicitly trained on.

## Models

The project implements the following models:

*   **Siamese Vision Transformer (SiameseViT):** A Siamese network based on the Vision Transformer architecture (implemented in `model/siamvit.py`). This model learns to compare pairs of images and determine their similarity.
*   **ResNet50 Multi-Class Classifier:** A ResNet50 model finetuned for multi-class classification of artists (implemented in `model/cnn_finetune.py`). This model learns to classify artworks based on their artist.
*   **Siamese ResNet50:** A Siamese network based on the ResNet50 architecture (implemented in `model/siamese_resnet.py`). This model learns to compare pairs of images and determine their similarity.
*   **Vision Transformer (ViT) Multi-Class Classifier:** A Vision Transformer model finetuned for multi-class classification of artists (implemented in `model/vit_finetune.py`). This model learns to classify artworks based on their artist.
*   **Siamese ArtNet:** A Siamese network with adaptive triplet loss (implemented in `model/siameseAdaptative.py`). This model learns to compare pairs of images and determine their similarity.

## Dataset

The project uses the WikiArt dataset, which contains a large collection of artworks from various artists and styles.

## Data Preparation

The data preparation pipeline involves the following steps:

*   Loading the WikiArt dataset.
*   Preprocessing the images (resizing, normalization, data augmentation).
*   Creating data loaders for training and evaluation.

## Training

The training process involves the following steps:

1.  Loading the dataset.
2.  Loading the model.
3.  Setting up the optimizer and loss function.
4.  Training the model for a specified number of epochs.

To train the models, run the following scripts:

*   `train_siamese_resnet.py`: Trains the Siamese ResNet50 model.
*   `train_cnn.py`: Trains the ResNet50 multi-class classifier.
*   `train_vit.py`: Trains the ViT multi-class classifier.
*   `train_adapvit.py`: Trains the Siamese ArtNet model with adaptive triplet loss, using the `scripts/main_adapvit.py` script.

## Inference

The `inference.py` script can be used to compute embeddings for images and calculate the similarity between them.

To use the `inference.py` script:

1.  Load the pre-trained model.
2.  Preprocess the images.
3.  Compute the embeddings for the images.
4.  Calculate the similarity between the embeddings.

## Embedding Similarity

The `embeding_similarity.py` script can be used to generate embeddings for images, store them in a Faiss index, and search for similar images.

To use the `embeding_similarity.py` script:

1.  Create an `EmbeddingDatabase` object.
2.  Generate embeddings for all images in a folder.
3.  Save the index for future use (optional).
4.  Search for similar images to a query image.

## Evaluation

The models are evaluated using the following metrics:

*   Few-Shot Accuracy (N-Shot-K-Way)
*   Mean Average Precision (mAP@K)
*   Few-Shot Generalization Gap

## Future Work

*   Explore different architectures and training strategies.
*   Investigate the use of different datasets.
*   Develop a more robust and scalable system for artist attribution.
