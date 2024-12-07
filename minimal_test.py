import streamlit as st
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import torch
from google.cloud import storage, firestore
import os, uuid
from datetime import datetime, timezone
import faiss
import numpy as np

st.title("Test with GCS + Firestore + FAISS")

@st.cache_resource
def load_models():
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return clip_model, clip_processor

clip_model, clip_processor = load_models()

storage_client = storage.Client()
firestore_client = firestore.Client()
raw_data_bucket = 'mcg-raw-data'

# Simple in-memory FAISS index tracking
index = None
id_to_query_id = []

def store_embedding_in_faiss(query_id, embedding_vector):
    global index, id_to_query_id
    embedding_vector = np.asarray(embedding_vector, dtype='float32')
    if index is None:
        embedding_dimension = embedding_vector.shape[0]
        index = faiss.IndexFlatL2(embedding_dimension)
    index.add(np.array([embedding_vector]))
    id_to_query_id.append(query_id)
    embedding_id = len(id_to_query_id) - 1
    return embedding_id

image_file = st.file_uploader("Upload a small image", type=["png", "jpg", "jpeg"])

if image_file is not None:
    query_id = str(uuid.uuid4())
    st.write("Query ID:", query_id)

    # GCS Upload
    st.write("Before GCS upload...")
    image_file.seek(0)
    bucket = storage_client.bucket(raw_data_bucket)
    blob = bucket.blob(f"images/{query_id}.jpg")
    blob.upload_from_file(image_file)
    st.write("After GCS upload.")

    data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_id": "test_user",
        "modality": "image",
        "raw_data_path": f"gs://{raw_data_bucket}/images/{query_id}.jpg",
        "metadata": {}
    }

    # Firestore write
    st.write("Before Firestore write...")
    doc_ref = firestore_client.collection("queries").document(query_id)
    doc_ref.set(data)
    st.write("After Firestore write.")

    # Embedding Generation + FAISS
    with st.spinner("Generating embedding..."):
        image_file.seek(0)
        image = Image.open(image_file).convert("RGB")
        image = image.resize((224, 224))
        with torch.no_grad():
            inputs = clip_processor(images=image, return_tensors="pt")
            outputs = clip_model.get_image_features(**inputs)
        embedding_vector = outputs.detach().numpy().flatten()

        # Normalize
        embedding_vector /= np.linalg.norm(embedding_vector)

        st.write("Before FAISS indexing...")
        embedding_id = store_embedding_in_faiss(query_id, embedding_vector)
        st.write("After FAISS indexing. Embedding ID:", embedding_id)

        doc_ref.update({'embedding_id': embedding_id})

    st.success("Done.")
