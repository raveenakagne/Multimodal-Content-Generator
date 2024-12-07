import streamlit as st
from google.cloud import storage, firestore
import uuid
from datetime import datetime
import os
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
from PIL import Image
import torch
import time

# Initialize GCP clients
storage_client = storage.Client()
firestore_client = firestore.Client()

# Set GCP bucket names
raw_data_bucket = 'mcg-raw-data'
embedding_data_bucket = 'mcg-embeddings'

# Load Models
@st.cache_resource
def load_models():
    st.write("Loading models...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    text_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    st.write("Models loaded successfully.")
    return clip_model, clip_processor, text_embedding_model

clip_model, clip_processor, text_embedding_model = load_models()

# Streamlit UI
st.title("Multimodal Content Generation Input")

# Session state for query ID
if 'query_id' not in st.session_state:
    st.session_state['query_id'] = None

# User ID (optional)
user_id = st.text_input("User ID (optional)", value="test_user")

# Modality Selection
modality = st.selectbox("Select Input Modality", ["Text", "Image"])

if modality == "Text":
    text_input = st.text_area("Enter your text query")
elif modality == "Image":
    image_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if st.button("Submit"):
    query_id = str(uuid.uuid4())
    st.session_state['query_id'] = query_id
    timestamp = datetime.utcnow()
    data = {
        "timestamp": timestamp.isoformat(),
        "user_id": user_id,
        "modality": modality.lower(),
        "raw_data_path": "",
        "metadata": {}
    }

    os.makedirs('./temp', exist_ok=True)  # Ensure temp directory exists

    if modality == "Text" and text_input:
        st.write("Processing text...")
        local_text_path = f"./temp/{query_id}.txt"
        with open(local_text_path, 'w') as f:
            f.write(text_input)

        bucket = storage_client.bucket(raw_data_bucket)
        blob = bucket.blob(f"text/{query_id}.txt")
        blob.upload_from_filename(local_text_path)

        data["raw_data_path"] = f"gs://{raw_data_bucket}/text/{query_id}.txt"
        data["metadata"]["text"] = text_input
        os.remove(local_text_path)

        doc_ref = firestore_client.collection("queries").document(query_id)
        doc_ref.set(data)

        st.success("Your input has been submitted successfully!")
        st.write("Query ID:", query_id)
        st.write("Raw Data Path:", data["raw_data_path"])

        st.info("Generating embedding for text...")
        embedding_vector = text_embedding_model.encode(text_input)
        embedding_vector /= np.linalg.norm(embedding_vector)

        local_embedding_path = f"./temp/{query_id}_embedding.npy"
        np.save(local_embedding_path, embedding_vector)
        embedding_bucket = storage_client.bucket(embedding_data_bucket)
        embedding_blob = embedding_bucket.blob(f"text/{query_id}_embedding.npy")
        embedding_blob.upload_from_filename(local_embedding_path)

        doc_ref.update({'embedding_path': f"gs://{embedding_data_bucket}/text/{query_id}_embedding.npy"})
        os.remove(local_embedding_path)

        st.success("Embedding generated and stored.")

    elif modality == "Image" and image_file:
        st.write("Processing image...")
        image_file.seek(0)
        bucket = storage_client.bucket(raw_data_bucket)
        blob = bucket.blob(f"images/{query_id}.jpg")
        blob.upload_from_file(image_file, content_type=image_file.type)

        data["raw_data_path"] = f"gs://{raw_data_bucket}/images/{query_id}.jpg"

        doc_ref = firestore_client.collection("queries").document(query_id)
        doc_ref.set(data)

        st.success("Your input has been submitted successfully!")
        st.write("Query ID:", query_id)
        st.write("Raw Data Path:", data["raw_data_path"])

        st.info("Generating embedding for image...")
        image_file.seek(0)
        image = Image.open(image_file).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = clip_model.get_image_features(**inputs)
        embedding_vector = outputs.detach().numpy().flatten()
        embedding_vector /= np.linalg.norm(embedding_vector)

        local_embedding_path = f"./temp/{query_id}_embedding.npy"
        np.save(local_embedding_path, embedding_vector)
        embedding_bucket = storage_client.bucket(embedding_data_bucket)
        embedding_blob = embedding_bucket.blob(f"images/{query_id}_embedding.npy")
        embedding_blob.upload_from_filename(local_embedding_path)

        doc_ref.update({'embedding_path': f"gs://{embedding_data_bucket}/images/{query_id}_embedding.npy"})
        os.remove(local_embedding_path)

        st.success("Embedding generated and stored.")

    else:
        st.error("Please provide valid input.")
        st.stop()

if st.session_state['query_id']:
    query_id = st.session_state['query_id']
    st.header("Query Results")
    doc_ref = firestore_client.collection("queries").document(query_id)
    doc = doc_ref.get()

    if doc.exists:
        doc_data = doc.to_dict()
        st.json(doc_data)
    else:
        st.write("Document not found in Firestore.")
