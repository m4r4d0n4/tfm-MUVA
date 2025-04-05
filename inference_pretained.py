import torch
from torchvision import transforms
from PIL import Image
from model.siamvit import SiameseViT  # Importa tu modelo correctamente

# Configuración del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar el modelo
model = SiameseViT().to(device)
model.load_state_dict(torch.load('siamese_vit_wikiart.pth', map_location=device))
model.eval()

# Transformaciones iguales a las del dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Lista de imágenes a probar
image_paths = ["./wikiart_images/test.png", "./wikiart_images/test2.jpg", "./wikiart_images/test3.jpg","./wikiart_images/toni.jpeg"]

# Función para cargar y preprocesar una imagen
def process_image(image_path):
    image = Image.open(image_path).convert("RGB")  # Asegura que la imagen sea RGB
    image = transform(image).unsqueeze(0).to(device)  # Agrega la dimensión de batch
    return image

embeddings= []
# Inferencia
for i, image_path in enumerate(image_paths):
    image = process_image(image_path)
    with torch.no_grad():
        embedding = model.forward_one(image)  # Usa el método para una sola imagen  # Ajusta esta línea si el modelo espera una estructura diferente
        embeddings.append(embedding)
    print(f"Embedding para {image_path}:")
    print(embedding.cpu().numpy())  # Convertir a numpy para visualizar mejor
    print("=" * 50)

# Convertir la lista de embeddings a un tensor
embeddings_tensor = torch.stack(embeddings).squeeze()  # Formato: (N, 128)

# Calcular matriz de distancias euclidianas
euclidean_distances = torch.cdist(embeddings_tensor, embeddings_tensor, p=2)

# Calcular similitud coseno (equivale al producto punto si los embeddings están normalizados)
cosine_similarities = torch.mm(embeddings_tensor, embeddings_tensor.T)  # Producto punto

# Imprimir resultados
print("\nDistancias Euclidianas entre embeddings:")
print(euclidean_distances.cpu().numpy())

print("\nSimilitud Coseno entre embeddings:")
print(cosine_similarities.cpu().numpy())