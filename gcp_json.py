import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import matplotlib.pyplot as plt
import json
from google.cloud import storage
import os

from google.cloud import storage

client = storage.Client()
print("Project ID:", client.project)


# Set device to MPS if available, else CPU
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Load BLIP model and processor
model_id = "Salesforce/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(model_id)
model = BlipForConditionalGeneration.from_pretrained(model_id).to(device)

def generate_image_caption(image_path):
    """
    Generates a caption for an image using the BLIP model.
    
    Parameters:
    - image_path (str): The path to the image file.

    Returns:
    - dict: JSON containing the image path and generated caption.
    """
    # Load and process the image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    # Generate caption
    with torch.no_grad():
        output = model.generate(**inputs, max_length=50, num_beams=5)
    caption = processor.decode(output[0], skip_special_tokens=True)
    
    # Create JSON object with image path and caption
    result = {
        "image_path": image_path,
        "caption": caption
    }
    
    return result

def upload_json_to_gcs(bucket_name, destination_blob_name, json_data):
    """
    Uploads a JSON object directly to Google Cloud Storage.
    
    Parameters:
    - bucket_name (str): The name of the GCP bucket.
    - destination_blob_name (str): The desired name of the file in the bucket.
    - json_data (dict): The JSON data to be uploaded.
    """
    # Initialize GCS client and specify the bucket
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    
    # Convert JSON object to string and upload
    blob.upload_from_string(
        data=json.dumps(json_data),
        content_type="application/json"
    )
    print(f"JSON object uploaded to {bucket_name}/{destination_blob_name}")

def display_captioned_image(image_path, bucket_name):
    """
    Displays the image alongside the generated caption and uploads caption JSON to GCP.
    
    Parameters:
    - image_path (str): The path to the image file.
    - bucket_name (str): The GCP bucket name.
    """
    # Generate caption and get JSON result
    result = generate_image_caption(image_path)
    caption = result["caption"]
    image = Image.open(image_path).convert("RGB")
    
    # Display image and caption
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.title('Original Image')
    
    plt.subplot(1, 2, 2)
    plt.text(0.05, 0.5, caption, wrap=True, fontsize=10)
    plt.axis('off')
    plt.title('Generated Caption')
    
    plt.tight_layout()
    plt.show()
    
    print("Image Description:")
    print(caption)
    print("-" * 100)
    
    
    image_name = os.path.basename(image_path)
    destination_blob_name = f"{os.path.splitext(image_name)[0]}.json"
    
    # Upload - GCS
    upload_json_to_gcs(bucket_name, destination_blob_name, result)


image_path = "example_flicker8kimages/996712323.jpg"
bucket_name = "user-queries-bucket-1"
display_captioned_image(image_path, bucket_name)
