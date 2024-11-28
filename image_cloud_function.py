# Assuming you have installed required packages
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from google.cloud import storage, firestore
import os

# Initialize GCP clients
storage_client = storage.Client()
firestore_client = firestore.Client()

# Load BLIP model and processor
model_id = "Salesforce/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(model_id)
model = BlipForConditionalGeneration.from_pretrained(model_id).to("cpu")

def process_image(query_id):
    # Get raw data path from Firestore
    doc_ref = firestore_client.collection("queries").document(query_id)
    doc = doc_ref.get()
    if not doc.exists:
        print(f"No document found for query_id: {query_id}")
        return

    data = doc.to_dict()
    raw_data_path = data["raw_data_path"]

    # Download image from GCS
    bucket_name = "mcg-raw-data"
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(f"images/{query_id}.jpg")
    local_image_path = f"/tmp/{query_id}.jpg"
    blob.download_to_filename(local_image_path)

    # Generate caption
    image = Image.open(local_image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)

    # Update Firestore with caption
    doc_ref.update({
        "metadata.caption": caption
    })

    # Clean up local file
    os.remove(local_image_path)
