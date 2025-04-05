import torch
import numpy as np
from tqdm import tqdm
import argparse

from dataset.cnn_dataset import WikiArtCNNDataset
from dataset.dataset import WikiArtTripletDataset
from inference import load_model, preprocess_image, compute_image_embedding

def create_embedding_database(model_type, model_path, output_file):
    """
    Creates an embedding database for the WikiArt dataset.
    
    Args:
        model_type (str): Type of model to use ('siamese_vit', 'resnet50', 'siamese_resnet')
        model_path (str): Path to the saved model weights
        output_file (str): Path to save the embedding database
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model
    model = load_model(model_type=model_type, model_path=model_path).to(device)
    model.eval()
    
    # Load the dataset
    if model_type == 'resnet50':
        dataset = WikiArtCNNDataset(split="train")
    elif model_type == 'vit_finetune':
        dataset = WikiArtCNNDataset(split="train")
    elif model_type == 'siamese_vit' or model_type == 'siamese_resnet':
        dataset = WikiArtTripletDataset(split="train", siamese=True)
    else:
        raise ValueError(f"Invalid model_type: {model_type}")
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # Create a list to store the embeddings, artist names, and image IDs
    embedding_list = []
    artist_list = []
    image_id_list = []
    
    # Iterate over the dataset and compute the embeddings
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            if model_type == 'siamese_vit' or model_type == 'siamese_resnet':
                image = data['anchor'].to(device)
            else:
                image, label = data
                image = image.to(device)
            
            embedding = compute_image_embedding(model, image, model_type)
            
            embedding_list.append(embedding)
            
            if model_type == 'siamese_vit' or model_type == 'siamese_resnet':
                artist_list.append(data['artist'][0])
            else:
                artist_list.append(dataset.artists[data[1].item()])
            
            image_id_list.append(i)
    
    # Convert the lists to NumPy arrays
    embedding_array = np.concatenate(embedding_list, axis=0)
    artist_array = np.array(artist_list)
    image_id_array = np.array(image_id_list)
    
    # Save the embedding database to a file
    np.savez(output_file, embeddings=embedding_array, artists=artist_array, image_ids=image_id_array)
    
    print(f"Embedding database saved to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create an embedding database for the WikiArt dataset.')
    parser.add_argument('--model_type', type=str, required=True, help="Type of model to use ('siamese_vit', 'resnet50', 'siamese_resnet')")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model weights')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the embedding database')
    
    args = parser.parse_args()
    
    create_embedding_database(args.model_type, args.model_path, args.output_file)
