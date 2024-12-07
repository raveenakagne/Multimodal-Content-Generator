import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import numpy as np
import os
import sys

def main():
    print("Script started...", flush=True)

    # Check if test.jpg exists
    image_path = "/Users/spartan/Downloads/Projects/Multimodal-Content-Generator/example_flicker8kimages/9950858.jpg"
    if not os.path.exists(image_path):
        print(f"ERROR: {image_path} not found in {os.getcwd()}", flush=True)
        sys.exit(1)

    print("Loading model...", flush=True)
    try:
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_model.eval()
    except Exception as e:
        print(f"Error loading CLIP model: {e}", flush=True)
        sys.exit(1)

    print("Loading processor...", flush=True)
    try:
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    except Exception as e:
        print(f"Error loading CLIPProcessor: {e}", flush=True)
        sys.exit(1)

    print(f"Loading image from {image_path}...", flush=True)
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image: {e}", flush=True)
        sys.exit(1)

    # Resize the image to reduce computation
    print("Resizing image...", flush=True)
    image = image.resize((224, 224))

    print("Generating embeddings...", flush=True)
    try:
        with torch.no_grad():
            inputs = clip_processor(images=image, return_tensors="pt")
            outputs = clip_model.get_image_features(**inputs)
        embedding_vector = outputs.detach().numpy().flatten()
        embedding_vector = embedding_vector / np.linalg.norm(embedding_vector)
    except Exception as e:
        print(f"Error generating embeddings: {e}", flush=True)
        sys.exit(1)

    print("Embedding vector generated successfully!", flush=True)
    print("Embedding vector:", embedding_vector, flush=True)
    print("Embedding vector shape:", embedding_vector.shape, flush=True)

    print("Script ended successfully!", flush=True)

if __name__ == "__main__":
    main()
