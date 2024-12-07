import streamlit as st
from google.cloud import storage, firestore
import uuid
from datetime import datetime
import os
import numpy as np
import faiss
import json
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
from PIL import Image
import torch

# Initialize GCP clients
storage_client = storage.Client()
firestore_client = firestore.Client()

# Set GCP bucket names
raw_data_bucket = 'mcg-raw-data'

# Initialize FAISS index and variables
index = None
id_to_query_id = []

# Initialize models
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

text_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to load FAISS index
def load_faiss_index():
    global index, id_to_query_id
    if os.path.exists('faiss_index.index'):
        index = faiss.read_index('faiss_index.index')
        id_to_query_id = np.load('id_to_query_id.npy', allow_pickle=True).tolist()
    else:
        index = None
        id_to_query_id = []
        st.warning("FAISS index not found. It will be created.")

# Function to save FAISS index
def save_faiss_index():
    faiss.write_index(index, 'faiss_index.index')
    # Save the id_to_query_id mapping
    with open('id_to_query_id.npy', 'wb') as f:
        np.save(f, np.array(id_to_query_id))

# Ensure temporary directory exists
def ensure_temp_dir():
    temp_dir = './temp'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    return temp_dir

# Store embedding in FAISS
def store_embedding_in_faiss(query_id, embedding_vector):
    global index, id_to_query_id

    # Convert embedding_vector to float32 if necessary
    embedding_vector = np.asarray(embedding_vector, dtype='float32')

    # Initialize FAISS index if not already done
    if index is None:
        embedding_dimension = embedding_vector.shape[0]
        index = faiss.IndexFlatL2(embedding_dimension)  # L2 distance

    # Add embedding to the FAISS index
    index.add(np.array([embedding_vector]))

    # Keep track of the query_id
    id_to_query_id.append(query_id)

    # Return the index of the added embedding as embedding_id
    embedding_id = len(id_to_query_id) - 1  # Zero-based indexing

    return embedding_id

# Call the function at the start
load_faiss_index()

# Streamlit UI
st.title("Multimodal Content Generation Input")

# Session state to retain query ID
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

# Submit Button
if st.button("Submit"):
    query_id = str(uuid.uuid4())
    st.session_state['query_id'] = query_id  # Save query ID in session state
    timestamp = datetime.utcnow()
    data = {
        "timestamp": timestamp.isoformat(),
        "user_id": user_id,
        "modality": modality.lower(),
        "raw_data_path": "",
        "metadata": {}
    }

    ensure_temp_dir()  # Ensure temp directory exists

    if modality == "Text" and text_input:
        # Save text to a local file
        local_text_path = f"./temp/{query_id}.txt"
        with open(local_text_path, 'w') as f:
            f.write(text_input)

        # Upload text file to GCS
        bucket = storage_client.bucket(raw_data_bucket)
        blob = bucket.blob(f"text/{query_id}.txt")
        blob.upload_from_filename(local_text_path)

        data["raw_data_path"] = f"gs://{raw_data_bucket}/text/{query_id}.txt"

        # Remove the local file
        os.remove(local_text_path)

        # Also save the text in Firestore metadata
        data["metadata"]["text"] = text_input

    elif modality == "Image" and image_file:
        # Save image to a local file
        local_image_path = f"./temp/{query_id}.jpg"
        with open(local_image_path, 'wb') as f:
            f.write(image_file.read())

        # Upload image to GCS
        bucket = storage_client.bucket(raw_data_bucket)
        blob = bucket.blob(f"images/{query_id}.jpg")
        blob.upload_from_filename(local_image_path)

        data["raw_data_path"] = f"gs://{raw_data_bucket}/images/{query_id}.jpg"

        # Remove the local file
        os.remove(local_image_path)

    else:
        st.error("Please provide valid input.")
        st.stop()

    # Save data to Firestore
    doc_ref = firestore_client.collection("queries").document(query_id)
    doc_ref.set(data)

    st.success("Your input has been submitted successfully!")
    st.write("Query ID:", query_id)
    st.write("Raw Data Path:", data["raw_data_path"])

    # Generate embedding immediately
    st.info("Generating embedding...")
    if modality == "Text":
        # Generate embedding
        embedding_vector = text_embedding_model.encode(text_input)
        embedding_vector /= np.linalg.norm(embedding_vector)

        # Store embedding in FAISS
        embedding_id = store_embedding_in_faiss(query_id, embedding_vector)

        # Update Firestore
        doc_ref.update({'embedding_id': embedding_id})

        # Save FAISS index
        save_faiss_index()

    elif modality == "Image":
        # Load and preprocess the image
        image = Image.open(local_image_path).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = clip_model.get_image_features(**inputs)
        embedding_vector = outputs.detach().numpy().flatten()
        embedding_vector /= np.linalg.norm(embedding_vector)

        # Store embedding in FAISS
        embedding_id = store_embedding_in_faiss(query_id, embedding_vector)

        # Update Firestore
        doc_ref.update({'embedding_id': embedding_id})

        # Save FAISS index
        save_faiss_index()

    st.success("Embedding generated and stored.")

# Display query results if query_id is available
if st.session_state['query_id']:
    query_id = st.session_state['query_id']
    st.header("Query Results")
    st.write("Query ID:", query_id)

    # Retrieve the document from Firestore
    doc_ref = firestore_client.collection("queries").document(query_id)
    doc = doc_ref.get()
    if doc.exists:
        doc_data = doc.to_dict()
        st.write("Firestore Document:")
        st.json(doc_data)

        # Display preprocessed data path (if available)
        preprocessed_data_path = doc_data.get("preprocessed_data_path", "Not available")
        st.write("Preprocessed Data Path:", preprocessed_data_path)

        # Display embedding ID
        embedding_id = doc_data.get("embedding_id", None)
        st.write("Embedding ID:", embedding_id)

        # Display embedding vector
        if embedding_id is not None and index is not None:
            try:
                embedding_vector = index.reconstruct(int(embedding_id))
                st.write("Embedding Vector:")
                st.write(embedding_vector)
            except Exception as e:
                st.write("Error retrieving embedding vector:", e)
        else:
            st.write("Embedding not available yet.")
    else:
        st.write("Document not found in Firestore.")
