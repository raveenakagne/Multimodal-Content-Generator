import streamlit as st
from google.cloud import storage, firestore
import uuid
from datetime import datetime
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image
import torch
import chromadb
from chromadb.config import Settings
from transformers import CLIPProcessor, CLIPModel
import PyPDF2  # For PDF processing

# Initialize GCP clients
storage_client = storage.Client()
firestore_client = firestore.Client()

# Set GCP bucket names
raw_data_bucket = 'mcg-raw-data'       # Correct bucket name
embedding_data_bucket = 'mcg-embeddings'

# Initialize Chroma DB
@st.cache_resource
def initialize_chroma():
    st.write("Initializing Chroma DB...")
    try:
        client = chromadb.Client(
            Settings(
                persist_directory="./chroma_db"    # Directory to store Chroma data
            )
        )
        collection = client.get_or_create_collection("audio_embeddings")
        st.write("Chroma DB initialized.")
        return collection
    except ValueError as ve:
        st.error(f"ChromaDB initialization error: {ve}")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error during ChromaDB initialization: {e}")
        st.stop()

chroma_collection = initialize_chroma()

# Load Models
@st.cache_resource
def load_models():
    st.write("Loading models...")
    try:
        # Text Embedding Model
        text_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # CLIP Models for Image Processing
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        st.write("Models loaded successfully.")
        return text_embedding_model, clip_model, clip_processor
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        st.stop()

# Correctly unpack the returned models
text_embedding_model, clip_model, clip_processor = load_models()

# Streamlit UI
st.title("Multimodal Content Generation Input")

# Session state for query ID
if 'query_id' not in st.session_state:
    st.session_state['query_id'] = None

# User ID (optional)
user_id = st.text_input("User ID (optional)", value="test_user")

# Modality Selection
modality = st.selectbox("Select Input Modality", ["Text", "Image", "Audio", "PDF"])

# Initialize variables based on modality
if modality == "Text":
    text_input = st.text_area("Enter your text query")
elif modality == "Image":
    image_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
elif modality == "Audio":
    audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a", "ogg"])
elif modality == "PDF":
    pdf_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if st.button("Submit"):
    # Check if the appropriate input is provided
    if (
        (modality == "Text" and text_input) or
        (modality == "Image" and image_file) or
        (modality == "Audio" and audio_file) or
        (modality == "PDF" and pdf_file)
    ):
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

            # Upload text to GCS
            try:
                bucket = storage_client.bucket(raw_data_bucket)
                blob = bucket.blob(f"text/{query_id}.txt")
                blob.upload_from_filename(local_text_path)
                st.write("Text uploaded to GCS.")
            except Exception as e:
                st.error(f"Failed to upload text: {e}")
                os.remove(local_text_path)
                st.stop()

            data["raw_data_path"] = f"gs://{raw_data_bucket}/text/{query_id}.txt"
            data["metadata"]["text"] = text_input
            os.remove(local_text_path)

            # Generate and store embedding
            st.info("Generating embedding for text...")
            try:
                embedding_vector = text_embedding_model.encode(text_input)
                embedding_vector /= np.linalg.norm(embedding_vector)
            except Exception as e:
                st.error(f"Failed to generate embedding: {e}")
                st.stop()

            # Save embedding as .npy locally and upload to GCS
            local_embedding_path = f"./temp/{query_id}_embedding.npy"
            try:
                np.save(local_embedding_path, embedding_vector)
                embedding_bucket = storage_client.bucket(embedding_data_bucket)
                embedding_blob = embedding_bucket.blob(f"text/{query_id}_embedding.npy")
                embedding_blob.upload_from_filename(local_embedding_path)
                st.write("Text embedding uploaded to GCS.")
            except Exception as e:
                st.error(f"Failed to upload embedding: {e}")
                os.remove(local_embedding_path)
                st.stop()

            data["embedding_path"] = f"gs://{embedding_data_bucket}/text/{query_id}_embedding.npy"
            os.remove(local_embedding_path)

            # Optional: Add embedding to Chroma DB (if applicable for text embeddings)
            # Example:
            # try:
            #     chroma_collection.add(
            #         documents=[text_input],
            #         embeddings=[embedding_vector.tolist()],
            #         ids=[query_id],
            #         metadatas=[{
            #             "raw_data_path": data["raw_data_path"],
            #             "user_id": user_id,
            #             "timestamp": data["timestamp"]
            #         }]
            #     )
            #     st.write("Embedding added to Chroma DB.")
            # except Exception as e:
            #     st.error(f"Failed to add embedding to Chroma DB: {e}")

            # Save metadata to Firestore
            try:
                doc_ref = firestore_client.collection("queries").document(query_id)
                doc_ref.set(data)
                st.write("Metadata saved to Firestore.")
            except Exception as e:
                st.error(f"Failed to save metadata to Firestore: {e}")
                st.stop()

            st.success("Your text input and embedding have been uploaded successfully!")
            st.write("Query ID:", query_id)
            st.write("Raw Data Path:", data["raw_data_path"])
            st.write("Embedding Path:", data["embedding_path"])

        elif modality == "Image" and image_file:
            st.write("Processing image...")
            image_file.seek(0)
            file_extension = image_file.type.split('/')[-1]  # e.g., 'jpeg', 'png'
            try:
                bucket = storage_client.bucket(raw_data_bucket)
                blob = bucket.blob(f"images/{query_id}.{file_extension}")
                blob.upload_from_file(image_file, content_type=image_file.type)
                st.write("Image uploaded to GCS.")
            except Exception as e:
                st.error(f"Failed to upload image: {e}")
                st.stop()

            data["raw_data_path"] = f"gs://{raw_data_bucket}/images/{query_id}.{file_extension}"

            # Generate and store embedding
            st.info("Generating embedding for image...")
            try:
                image_file.seek(0)
                image = Image.open(image_file).convert("RGB")
                inputs = clip_processor(images=image, return_tensors="pt")
                with torch.no_grad():
                    outputs = clip_model.get_image_features(**inputs)
                embedding_vector = outputs.detach().numpy().flatten()
                embedding_vector /= np.linalg.norm(embedding_vector)
            except Exception as e:
                st.error(f"Failed to generate embedding: {e}")
                st.stop()

            # Save embedding as .npy locally and upload to GCS
            local_embedding_path = f"./temp/{query_id}_embedding.npy"
            try:
                np.save(local_embedding_path, embedding_vector)
                embedding_bucket = storage_client.bucket(embedding_data_bucket)
                embedding_blob = embedding_bucket.blob(f"images/{query_id}_embedding.npy")
                embedding_blob.upload_from_filename(local_embedding_path)
                st.write("Image embedding uploaded to GCS.")
            except Exception as e:
                st.error(f"Failed to upload embedding: {e}")
                os.remove(local_embedding_path)
                st.stop()

            data["embedding_path"] = f"gs://{embedding_data_bucket}/images/{query_id}_embedding.npy"
            os.remove(local_embedding_path)

            # # Add embedding to Chroma DB
            # st.write("Adding embedding to Chroma DB...")
            # try:
            #     chroma_collection.add(
            #         documents=[f"Image document {query_id}"],
            #         embeddings=[embedding_vector.tolist()],
            #         ids=[query_id],
            #         metadatas=[{
            #             "raw_data_path": data["raw_data_path"],
            #             "user_id": user_id,
            #             "timestamp": data["timestamp"]
            #         }]
            #     )
            #     st.write("Embedding added to Chroma DB.")
            # except Exception as e:
            #     st.error(f"Failed to add embedding to Chroma DB: {e}")

            # Save metadata to Firestore
            try:
                doc_ref = firestore_client.collection("queries").document(query_id)
                doc_ref.set(data)
                st.write("Metadata saved to Firestore.")
            except Exception as e:
                st.error(f"Failed to save metadata to Firestore: {e}")
                st.stop()

            st.success("Your image and embedding have been uploaded successfully!")
            st.write("Query ID:", query_id)
            st.write("Raw Data Path:", data["raw_data_path"])
            st.write("Embedding Path:", data["embedding_path"])

        elif modality == "Audio" and audio_file:
            st.write("Processing audio...")
            audio_file.seek(0)
            file_extension = audio_file.type.split('/')[-1]  # e.g., 'mp3', 'wav'
            try:
                bucket = storage_client.bucket(raw_data_bucket)
                blob = bucket.blob(f"audio/{query_id}.{file_extension}")
                blob.upload_from_file(audio_file, content_type=audio_file.type)
                st.write("Audio uploaded to GCS.")
            except Exception as e:
                st.error(f"Failed to upload audio: {e}")
                st.stop()

            data["raw_data_path"] = f"gs://{raw_data_bucket}/audio/{query_id}.{file_extension}"

            # Generate and store embedding using CLAP (placeholder)
            st.info("Generating embedding for audio...")
            try:
                # Placeholder: Replace with actual CLAP embedding generation
                # Example:
                # embedding_vector = clap_model.encode(local_audio_path)
                # For demonstration, using a random vector
                embedding_vector = np.random.rand(512).astype('float32')  # Placeholder
                embedding_vector /= np.linalg.norm(embedding_vector)
            except Exception as e:
                st.error(f"Failed to generate embedding: {e}")
                st.stop()

            # Save embedding as .npy locally and upload to GCS
            local_embedding_path = f"./temp/{query_id}_embedding.npy"
            try:
                np.save(local_embedding_path, embedding_vector)
                embedding_bucket = storage_client.bucket(embedding_data_bucket)
                embedding_blob = embedding_bucket.blob(f"audio/{query_id}_embedding.npy")
                embedding_blob.upload_from_filename(local_embedding_path)
                st.write("Audio embedding uploaded to GCS.")
            except Exception as e:
                st.error(f"Failed to upload embedding: {e}")
                os.remove(local_embedding_path)
                st.stop()

            data["embedding_path"] = f"gs://{embedding_data_bucket}/audio/{query_id}_embedding.npy"
            os.remove(local_embedding_path)

            # Add embedding to Chroma DB
            st.write("Adding embedding to Chroma DB...")
            try:
                chroma_collection.add(
                    documents=[f"Audio document {query_id}"],
                    embeddings=[embedding_vector.tolist()],
                    ids=[query_id],
                    metadatas=[{
                        "raw_data_path": data["raw_data_path"],
                        "user_id": user_id,
                        "timestamp": data["timestamp"]
                    }]
                )
                st.write("Embedding added to Chroma DB.")
            except Exception as e:
                st.error(f"Failed to add embedding to Chroma DB: {e}")

            # Save metadata to Firestore
            try:
                doc_ref = firestore_client.collection("queries").document(query_id)
                doc_ref.set(data)
                st.write("Metadata saved to Firestore.")
            except Exception as e:
                st.error(f"Failed to save metadata to Firestore: {e}")
                st.stop()

            st.success("Your audio and embedding have been uploaded successfully!")
            st.write("Query ID:", query_id)
            st.write("Raw Data Path:", data["raw_data_path"])
            st.write("Embedding Path:", data["embedding_path"])

        elif modality == "PDF" and pdf_file:
            st.write("Processing PDF...")
            pdf_file.seek(0)
            try:
                # Save PDF locally
                local_pdf_path = f"./temp/{query_id}.pdf"
                with open(local_pdf_path, 'wb') as f:
                    f.write(pdf_file.read())
                
                # Upload PDF to GCS
                bucket = storage_client.bucket(raw_data_bucket)
                blob = bucket.blob(f"pdf/{query_id}.pdf")
                blob.upload_from_filename(local_pdf_path)
                st.write("PDF uploaded to GCS.")
            except Exception as e:
                st.error(f"Failed to upload PDF: {e}")
                if os.path.exists(local_pdf_path):
                    os.remove(local_pdf_path)
                st.stop()

            data["raw_data_path"] = f"gs://{raw_data_bucket}/pdf/{query_id}.pdf"

            # Extract text from PDF
            st.info("Extracting text from PDF...")
            try:
                with open(local_pdf_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() or ""
            except Exception as e:
                st.error(f"Failed to extract text from PDF: {e}")
                os.remove(local_pdf_path)
                st.stop()

            os.remove(local_pdf_path)

            # Generate and store embedding
            st.info("Generating embedding for PDF text...")
            try:
                embedding_vector = text_embedding_model.encode(text)
                embedding_vector /= np.linalg.norm(embedding_vector)
            except Exception as e:
                st.error(f"Failed to generate embedding: {e}")
                st.stop()

            # Save embedding as .npy locally and upload to GCS
            local_embedding_path = f"./temp/{query_id}_embedding.npy"
            try:
                np.save(local_embedding_path, embedding_vector)
                embedding_bucket = storage_client.bucket(embedding_data_bucket)
                embedding_blob = embedding_bucket.blob(f"pdf/{query_id}_embedding.npy")
                embedding_blob.upload_from_filename(local_embedding_path)
                st.write("PDF embedding uploaded to GCS.")
            except Exception as e:
                st.error(f"Failed to upload embedding: {e}")
                os.remove(local_embedding_path)
                st.stop()

            data["embedding_path"] = f"gs://{embedding_data_bucket}/pdf/{query_id}_embedding.npy"
            os.remove(local_embedding_path)

            # # Add embedding to Chroma DB
            # st.write("Adding embedding to Chroma DB...")
            # try:
            #     chroma_collection.add(
            #         documents=[text],  # Using extracted text as the document
            #         embeddings=[embedding_vector.tolist()],
            #         ids=[query_id],
            #         metadatas=[{
            #             "raw_data_path": data["raw_data_path"],
            #             "user_id": user_id,
            #             "timestamp": data["timestamp"]
            #         }]
            #     )
            #     st.write("Embedding added to Chroma DB.")
            # except Exception as e:
            #     st.error(f"Failed to add embedding to Chroma DB: {e}")

            # Save metadata to Firestore
            try:
                doc_ref = firestore_client.collection("queries").document(query_id)
                doc_ref.set(data)
                st.write("Metadata saved to Firestore.")
            except Exception as e:
                st.error(f"Failed to save metadata to Firestore: {e}")
                st.stop()

            st.success("Your PDF and embedding have been uploaded successfully!")
            st.write("Query ID:", query_id)
            st.write("Raw Data Path:", data["raw_data_path"])
            st.write("Embedding Path:", data["embedding_path"])

    else:
        st.error("Please provide valid input.")
        st.stop()

    # Display query results
    if st.session_state['query_id']:
        query_id = st.session_state['query_id']
        st.header("Query Results")
        try:
            doc_ref = firestore_client.collection("queries").document(query_id)
            doc = doc_ref.get()

            if doc.exists:
                doc_data = doc.to_dict()
                st.json(doc_data)
            else:
                st.write("Document not found in Firestore.")
        except Exception as e:
            st.error(f"Failed to retrieve document: {e}")
