from datasets import load_dataset
import os
import random
from PIL import Image

def generate_few_shot_dataset(num_authors=20, support_size=10, query_size=5, dataset_name="thers2m/rijksmuseum_painting_dataset"):
    """
    Generates a few-shot dataset with the specified number of authors,
    support set size, and query set size.
    """

    # Load the dataset
    ds = load_dataset(dataset_name, split="train")

    # Get the list of authors
    authors = list(set(ds["artist"]))

    # Calculate author image counts
    author_image_counts = {}
    for author in authors:
        author_ds = ds.filter(lambda example: example["artist"] == author)
        author_image_counts[author] = len(list(author_ds))

    # Filter authors to include only those with at least support_size + query_size images
    min_images = support_size + query_size
    filtered_authors = [author for author, count in author_image_counts.items() if count >= min_images]
    random.shuffle(filtered_authors)

    # Select the first num_authors authors
    selected_authors = filtered_authors[:num_authors]

    # Create the directories
    for i, author in enumerate(selected_authors):
        author_dir = os.path.join("author_" + str(i + 1))
        support_dir = os.path.join(author_dir, "support")
        query_dir = os.path.join(author_dir, "query")

        os.makedirs(support_dir, exist_ok=True)
        os.makedirs(query_dir, exist_ok=True)

        # Filter the dataset by author
        author_ds = ds.filter(lambda example: example["artist"] == author)

        # Select support and query images
        author_images = [item['image'] for item in author_ds]
        random.shuffle(author_images)

        support_images = author_images[:support_size]
        query_images = author_images[support_size:support_size + query_size]

        # Save the images
        for j, image in enumerate(support_images):
            image.save(os.path.join(support_dir, f"support_{j}.jpg"))

        for j, image in enumerate(query_images):
            image.save(os.path.join(query_dir, f"query_{j}.jpg"))

if __name__ == "__main__":
    generate_few_shot_dataset()
