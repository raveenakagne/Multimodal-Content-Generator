import streamlit as st
from google.cloud import storage, firestore
import uuid
from datetime import datetime, timezone
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image
import torch
import chromadb
from chromadb.config import Settings
from transformers import CLIPProcessor, CLIPModel
import PyPDF2  # For PDF processing
import logging
from firecrawl import FirecrawlApp  # Firecrawl SDK
import requests
import time  # For implementing delays
import warnings

# Suppress specific warnings if necessary
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", message="Tried to instantiate class '__path__._path'")

# Configure logging
logging.basicConfig(
    filename='app.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logging.info("Streamlit app started.")

# Initialize GCP clients
try:
    storage_client = storage.Client()
    firestore_client = firestore.Client()
    logging.info("GCP clients initialized successfully.")
except Exception as e:
    st.error(f"Failed to initialize GCP clients: {e}")
    logging.error(f"Failed to initialize GCP clients: {e}", exc_info=True)
    st.stop()

# Initialize FirecrawlApp with API key
firecrawl_api_key = os.getenv('FIRECRAWL_API_KEY')
if not firecrawl_api_key:
    st.error("Firecrawl API key not found. Please set the FIRECRAWL_API_KEY environment variable.")
    logging.error("Firecrawl API key not found.")
    st.stop()

try:
    firecrawl_app = FirecrawlApp(api_key=firecrawl_api_key)
    logging.info("FirecrawlApp initialized successfully.")
except Exception as e:
    st.error(f"Failed to initialize FirecrawlApp: {e}")
    logging.error(f"Failed to initialize FirecrawlApp: {e}", exc_info=True)
    st.stop()

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
        logging.info("Chroma DB initialized successfully.")
        return collection
    except ValueError as ve:
        st.error(f"ChromaDB initialization error: {ve}")
        logging.error(f"ChromaDB initialization error: {ve}", exc_info=True)
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error during ChromaDB initialization: {e}")
        logging.error(f"Unexpected error during ChromaDB initialization: {e}", exc_info=True)
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
        logging.info("Models loaded successfully.")
        return text_embedding_model, clip_model, clip_processor
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        logging.error(f"Failed to load models: {e}", exc_info=True)
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
modality = st.selectbox("Select Input Modality", ["Text", "Image", "Audio", "PDF", "URL"])

# Initialize variables based on modality
if modality == "Text":
    text_input = st.text_area("Enter your text query")
elif modality == "Image":
    image_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
elif modality == "Audio":
    audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a", "ogg"])
elif modality == "PDF":
    pdf_file = st.file_uploader("Upload a PDF document", type=["pdf"])
elif modality == "URL":
    url_input = st.text_input("Enter the URL")

if st.button("Submit"):
    # Check if the appropriate input is provided
    if (
        (modality == "Text" and text_input) or
        (modality == "Image" and image_file) or
        (modality == "Audio" and audio_file) or
        (modality == "PDF" and pdf_file) or
        (modality == "URL" and url_input)
    ):
        query_id = str(uuid.uuid4())
        st.session_state['query_id'] = query_id
        timestamp = datetime.now(timezone.utc)  # Use timezone-aware datetime
        data = {
            "timestamp": timestamp.isoformat(),
            "user_id": user_id,
            "modality": modality.lower(),
            "raw_data_path": "",
            "embedding_path": "",
            "metadata": {}
        }

        os.makedirs('./temp', exist_ok=True)  # Ensure temp directory exists

        if modality == "Text" and text_input:
            st.write("Processing text...")
            local_text_path = f"./temp/{query_id}.txt"
            try:
                with open(local_text_path, 'w') as f:
                    f.write(text_input)
                logging.info(f"Text saved locally at {local_text_path}.")
            except Exception as e:
                st.error(f"Failed to save text locally: {e}")
                logging.error(f"Failed to save text locally: {e}", exc_info=True)
                st.stop()

            # Upload text to GCS
            try:
                bucket = storage_client.bucket(raw_data_bucket)
                blob = bucket.blob(f"text/{query_id}.txt")
                blob.upload_from_filename(local_text_path)
                st.write("Text uploaded to GCS.")
                logging.info(f"Text uploaded to gs://{raw_data_bucket}/text/{query_id}.txt")
            except Exception as e:
                st.error(f"Failed to upload text to GCS: {e}")
                logging.error(f"Failed to upload text to GCS: {e}", exc_info=True)
                os.remove(local_text_path)
                st.stop()

            data["raw_data_path"] = f"gs://{raw_data_bucket}/text/{query_id}.txt"
            data["metadata"]["text"] = text_input
            os.remove(local_text_path)
            logging.info(f"Local text file {local_text_path} removed after upload.")

            # Generate and store embedding
            st.info("Generating embedding for text...")
            try:
                embedding_vector = text_embedding_model.encode(text_input)
                embedding_vector /= np.linalg.norm(embedding_vector)
                logging.info("Text embedding generated successfully.")
            except Exception as e:
                st.error(f"Failed to generate embedding: {e}")
                logging.error(f"Failed to generate embedding: {e}", exc_info=True)
                st.stop()

            # Save embedding as .npy locally and upload to GCS
            local_embedding_path = f"./temp/{query_id}_embedding.npy"
            try:
                np.save(local_embedding_path, embedding_vector)
                embedding_bucket = storage_client.bucket(embedding_data_bucket)
                embedding_blob = embedding_bucket.blob(f"text/{query_id}_embedding.npy")
                embedding_blob.upload_from_filename(local_embedding_path)
                st.write("Text embedding uploaded to GCS.")
                logging.info(f"Text embedding uploaded to gs://{embedding_data_bucket}/text/{query_id}_embedding.npy")
            except Exception as e:
                st.error(f"Failed to upload embedding to GCS: {e}")
                logging.error(f"Failed to upload embedding to GCS: {e}", exc_info=True)
                os.remove(local_embedding_path)
                st.stop()

            data["embedding_path"] = f"gs://{embedding_data_bucket}/text/{query_id}_embedding.npy"
            os.remove(local_embedding_path)
            logging.info(f"Local embedding file {local_embedding_path} removed after upload.")

            # # Add embedding to Chroma DB
            # st.write("Adding embedding to Chroma DB...")
            # try:
            #     chroma_collection.add(
            #         documents=[text_input],
            #         embeddings=[embedding_vector.tolist()],
            #         ids=[query_id],
            #         metadatas=[{
            #             "raw_data_path": data["raw_data_path"],
            #             "user_id": user_id,
            #             "timestamp": data["timestamp"],
            #             "modality": modality.lower()
            #         }]
            #     )
            #     st.write("Embedding added to Chroma DB.")
            #     logging.info(f"Text embedding added to Chroma DB with ID {query_id}.")
            # except Exception as e:
            #     st.error(f"Failed to add embedding to Chroma DB: {e}")
            #     logging.error(f"Failed to add embedding to Chroma DB: {e}", exc_info=True)

            # Save metadata to Firestore
            try:
                doc_ref = firestore_client.collection("queries").document(query_id)
                doc_ref.set(data)
                st.write("Metadata saved to Firestore.")
                logging.info(f"Metadata for {query_id} saved to Firestore.")
            except Exception as e:
                st.error(f"Failed to save metadata to Firestore: {e}")
                logging.error(f"Failed to save metadata to Firestore: {e}", exc_info=True)
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
                logging.info(f"Image uploaded to gs://{raw_data_bucket}/images/{query_id}.{file_extension}")
            except Exception as e:
                st.error(f"Failed to upload image to GCS: {e}")
                logging.error(f"Failed to upload image to GCS: {e}", exc_info=True)
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
                logging.info("Image embedding generated successfully.")
            except Exception as e:
                st.error(f"Failed to generate embedding: {e}")
                logging.error(f"Failed to generate embedding: {e}", exc_info=True)
                st.stop()

            # Save embedding as .npy locally and upload to GCS
            local_embedding_path = f"./temp/{query_id}_embedding.npy"
            try:
                np.save(local_embedding_path, embedding_vector)
                embedding_bucket = storage_client.bucket(embedding_data_bucket)
                embedding_blob = embedding_bucket.blob(f"images/{query_id}_embedding.npy")
                embedding_blob.upload_from_filename(local_embedding_path)
                st.write("Image embedding uploaded to GCS.")
                logging.info(f"Image embedding uploaded to gs://{embedding_data_bucket}/images/{query_id}_embedding.npy")
            except Exception as e:
                st.error(f"Failed to upload embedding to GCS: {e}")
                logging.error(f"Failed to upload embedding to GCS: {e}", exc_info=True)
                os.remove(local_embedding_path)
                st.stop()

            data["embedding_path"] = f"gs://{embedding_data_bucket}/images/{query_id}_embedding.npy"
            os.remove(local_embedding_path)
            logging.info(f"Local embedding file {local_embedding_path} removed after upload.")

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
            #             "timestamp": data["timestamp"],
            #             "modality": modality.lower()
            #         }]
            #     )
            #     st.write("Embedding added to Chroma DB.")
            #     logging.info(f"Image embedding added to Chroma DB with ID {query_id}.")
            # except Exception as e:
            #     st.error(f"Failed to add embedding to Chroma DB: {e}")
            #     logging.error(f"Failed to add embedding to Chroma DB: {e}", exc_info=True)

            # Save metadata to Firestore
            try:
                doc_ref = firestore_client.collection("queries").document(query_id)
                doc_ref.set(data)
                st.write("Metadata saved to Firestore.")
                logging.info(f"Metadata for {query_id} saved to Firestore.")
            except Exception as e:
                st.error(f"Failed to save metadata to Firestore: {e}")
                logging.error(f"Failed to save metadata to Firestore: {e}", exc_info=True)
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
                logging.info(f"Audio uploaded to gs://{raw_data_bucket}/audio/{query_id}.{file_extension}")
            except Exception as e:
                st.error(f"Failed to upload audio to GCS: {e}")
                logging.error(f"Failed to upload audio to GCS: {e}", exc_info=True)
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
                logging.info("Audio embedding generated successfully (placeholder).")
            except Exception as e:
                st.error(f"Failed to generate embedding: {e}")
                logging.error(f"Failed to generate embedding: {e}", exc_info=True)
                st.stop()

            # Save embedding as .npy locally and upload to GCS
            local_embedding_path = f"./temp/{query_id}_embedding.npy"
            try:
                np.save(local_embedding_path, embedding_vector)
                embedding_bucket = storage_client.bucket(embedding_data_bucket)
                embedding_blob = embedding_bucket.blob(f"audio/{query_id}_embedding.npy")
                embedding_blob.upload_from_filename(local_embedding_path)
                st.write("Audio embedding uploaded to GCS.")
                logging.info(f"Audio embedding uploaded to gs://{embedding_data_bucket}/audio/{query_id}_embedding.npy")
            except Exception as e:
                st.error(f"Failed to upload embedding to GCS: {e}")
                logging.error(f"Failed to upload embedding to GCS: {e}", exc_info=True)
                os.remove(local_embedding_path)
                st.stop()

            data["embedding_path"] = f"gs://{embedding_data_bucket}/audio/{query_id}_embedding.npy"
            os.remove(local_embedding_path)
            logging.info(f"Local embedding file {local_embedding_path} removed after upload.")

            # # Add embedding to Chroma DB
            # st.write("Adding embedding to Chroma DB...")
            # try:
            #     chroma_collection.add(
            #         documents=[f"Audio document {query_id}"],
            #         embeddings=[embedding_vector.tolist()],
            #         ids=[query_id],
            #         metadatas=[{
            #             "raw_data_path": data["raw_data_path"],
            #             "user_id": user_id,
            #             "timestamp": data["timestamp"],
            #             "modality": modality.lower()
            #         }]
            #     )
            #     st.write("Embedding added to Chroma DB.")
            #     logging.info(f"Audio embedding added to Chroma DB with ID {query_id}.")
            # except Exception as e:
            #     st.error(f"Failed to add embedding to Chroma DB: {e}")
            #     logging.error(f"Failed to add embedding to Chroma DB: {e}", exc_info=True)

            # Save metadata to Firestore
            try:
                doc_ref = firestore_client.collection("queries").document(query_id)
                doc_ref.set(data)
                st.write("Metadata saved to Firestore.")
                logging.info(f"Metadata for {query_id} saved to Firestore.")
            except Exception as e:
                st.error(f"Failed to save metadata to Firestore: {e}")
                logging.error(f"Failed to save metadata to Firestore: {e}", exc_info=True)
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
                logging.info(f"PDF saved locally at {local_pdf_path}.")

                # Upload PDF to GCS
                bucket = storage_client.bucket(raw_data_bucket)
                blob = bucket.blob(f"pdf/{query_id}.pdf")
                blob.upload_from_filename(local_pdf_path)
                st.write("PDF uploaded to GCS.")
                logging.info(f"PDF uploaded to gs://{raw_data_bucket}/pdf/{query_id}.pdf")
            except Exception as e:
                st.error(f"Failed to upload PDF: {e}")
                if os.path.exists(local_pdf_path):
                    os.remove(local_pdf_path)
                logging.error(f"Failed to upload PDF: {e}", exc_info=True)
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
                logging.info("Text extracted from PDF successfully.")
            except Exception as e:
                st.error(f"Failed to extract text from PDF: {e}")
                logging.error(f"Failed to extract text from PDF: {e}", exc_info=True)
                os.remove(local_pdf_path)
                st.stop()

            os.remove(local_pdf_path)
            logging.info(f"Local PDF file {local_pdf_path} removed after upload.")

            # Generate and store embedding
            st.info("Generating embedding for PDF text...")
            try:
                embedding_vector = text_embedding_model.encode(text)
                embedding_vector /= np.linalg.norm(embedding_vector)
                logging.info("PDF text embedding generated successfully.")
            except Exception as e:
                st.error(f"Failed to generate embedding: {e}")
                logging.error(f"Failed to generate embedding: {e}", exc_info=True)
                st.stop()

            # Save embedding as .npy locally and upload to GCS
            local_embedding_path = f"./temp/{query_id}_embedding.npy"
            try:
                np.save(local_embedding_path, embedding_vector)
                embedding_bucket = storage_client.bucket(embedding_data_bucket)
                embedding_blob = embedding_bucket.blob(f"pdf/{query_id}_embedding.npy")
                embedding_blob.upload_from_filename(local_embedding_path)
                st.write("PDF embedding uploaded to GCS.")
                logging.info(f"PDF embedding uploaded to gs://{embedding_data_bucket}/pdf/{query_id}_embedding.npy")
            except Exception as e:
                st.error(f"Failed to upload embedding: {e}")
                logging.error(f"Failed to upload embedding: {e}", exc_info=True)
                os.remove(local_embedding_path)
                st.stop()

            data["embedding_path"] = f"gs://{embedding_data_bucket}/pdf/{query_id}_embedding.npy"
            os.remove(local_embedding_path)
            logging.info(f"Local embedding file {local_embedding_path} removed after upload.")

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
            #             "timestamp": data["timestamp"],
            #             "modality": modality.lower()
            #         }]
            #     )
            #     st.write("Embedding added to Chroma DB.")
            #     logging.info(f"PDF embedding added to Chroma DB with ID {query_id}.")
            # except Exception as e:
            #     st.error(f"Failed to add embedding to Chroma DB: {e}")
            #     logging.error(f"Failed to add embedding to Chroma DB: {e}", exc_info=True)

            # Save metadata to Firestore
            try:
                doc_ref = firestore_client.collection("queries").document(query_id)
                doc_ref.set(data)
                st.write("Metadata saved to Firestore.")
                logging.info(f"Metadata for {query_id} saved to Firestore.")
            except Exception as e:
                st.error(f"Failed to save metadata to Firestore: {e}")
                logging.error(f"Failed to save metadata to Firestore: {e}", exc_info=True)
                st.stop()

            st.success("Your PDF and embedding have been uploaded successfully!")
            st.write("Query ID:", query_id)
            st.write("Raw Data Path:", data["raw_data_path"])
            st.write("Embedding Path:", data["embedding_path"])

        elif modality == "URL" and url_input:
            st.write("Processing URL...")
            try:
                # Check URL accessibility first
                try:
                    response = requests.get(url_input, timeout=10)
                    if response.status_code != 200:
                        st.error(f"URL returned status code {response.status_code}.")
                        logging.error(f"URL returned status code {response.status_code}.")
                        st.stop()
                except Exception as e:
                    st.error(f"Failed to reach the URL: {e}")
                    logging.error(f"Failed to reach the URL: {e}", exc_info=True)
                    st.stop()

                # Use Firecrawl to scrape the URL
                st.info("Scraping the URL using Firecrawl...")
                with st.spinner('Scraping the URL using Firecrawl...'):
                    scrape_response = firecrawl_app.scrape_url(
                        url=url_input,
                        params={
                            'formats': ['markdown'],  # Using markdown for easy text extraction
                            'onlyMainContent': True,
                            'removeBase64Images': True
                        }
                    )

                # Log the entire response for debugging
                st.write("Scrape Response:", scrape_response)
                logging.info(f"Scrape Response: {scrape_response}")

                # if not scrape_response.get('success'):
                #     error_message = scrape_response.get('data', {}).get('error', 'Unknown error occurred.')
                #     st.error(f"Firecrawl failed to scrape the URL. Error: {error_message}")
                #     logging.error(f"Firecrawl failed to scrape the URL. Error: {error_message}")
                #     st.stop()

                scraped_data = scrape_response.get('data', {})
                markdown_content = scraped_data.get('markdown', '')
                html_content = scraped_data.get('html', '')
                links = scraped_data.get('links', [])
                images = scraped_data.get('screenshot', [])  # Assuming 'screenshot' contains image URLs

                # if not markdown_content and not html_content:
                #     st.warning("No extractable text found at the provided URL.")

                # # Save markdown or html locally
                # if markdown_content:
                #     local_content_path = f"./temp/{query_id}.md"
                #     try:
                #         with open(local_content_path, 'w', encoding='utf-8') as f:
                #             f.write(markdown_content)
                #         data["raw_data_path"] = f"gs://{raw_data_bucket}/url/{query_id}.md"
                #         logging.info(f"Scraped markdown content saved locally at {local_content_path}.")
                #     except Exception as e:
                #         st.error(f"Failed to save scraped markdown locally: {e}")
                #         logging.error(f"Failed to save scraped markdown locally: {e}", exc_info=True)
                #         st.stop()
                # elif html_content:
                #     local_content_path = f"./temp/{query_id}.html"
                #     try:
                #         with open(local_content_path, 'w', encoding='utf-8') as f:
                #             f.write(html_content)
                #         data["raw_data_path"] = f"gs://{raw_data_bucket}/url/{query_id}.html"
                #         logging.info(f"Scraped HTML content saved locally at {local_content_path}.")
                #     except Exception as e:
                #         st.error(f"Failed to save scraped HTML locally: {e}")
                #         logging.error(f"Failed to save scraped HTML locally: {e}", exc_info=True)
                #         st.stop()
                # else:
                #     st.warning("No content to save from the URL.")
                #     local_content_path = None

            except Exception as e:
                st.error(f"Failed to scrape the URL: {e}")
                logging.error(f"Failed to scrape the URL: {e}", exc_info=True)
                st.stop()

            # # Continue processing only if content was saved
            # if local_content_path:
            #     try:
            #         # Upload content to GCS
            #         bucket = storage_client.bucket(raw_data_bucket)
            #         if markdown_content:
            #             blob = bucket.blob(f"url/{query_id}.md")
            #         else:
            #             blob = bucket.blob(f"url/{query_id}.html")
            #         blob.upload_from_filename(local_content_path)
            #         st.write("Scraped content uploaded to GCS.")
            #         logging.info(f"Scraped content uploaded to gs://{raw_data_bucket}/url/{query_id}.{ 'md' if markdown_content else 'html' }")
            #     except Exception as e:
            #         st.error(f"Failed to upload scraped content to GCS: {e}")
            #         logging.error(f"Failed to upload scraped content to GCS: {e}", exc_info=True)
            #         os.remove(local_content_path)
            #         st.stop()

                # # Update metadata with scraped content
                # data["metadata"]["scraped_content"] = markdown_content if markdown_content else html_content

                # try:
                #     os.remove(local_content_path)
                #     logging.info(f"Local scraped content file {local_content_path} removed after upload.")
                # except Exception as e:
                #     logging.warning(f"Failed to remove local scraped content file {local_content_path}: {e}")

            # Generate and store embedding for text
            st.info("Generating embedding for scraped text...")
            try:
                text_to_embed = markdown_content if markdown_content else html_content
                if text_to_embed:
                    embedding_vector = text_embedding_model.encode(text_to_embed)
                    embedding_vector /= np.linalg.norm(embedding_vector)
                    logging.info("Scraped text embedding generated successfully.")
                else:
                    embedding_vector = None
                    logging.warning("No text to embed.")
            except Exception as e:
                st.error(f"Failed to generate embedding: {e}")
                logging.error(f"Failed to generate embedding: {e}", exc_info=True)
                st.stop()

            if embedding_vector is not None:
                # Save embedding as .npy locally and upload to GCS
                local_embedding_path = f"./temp/{query_id}_embedding.npy"
                try:
                    np.save(local_embedding_path, embedding_vector)
                    embedding_bucket = storage_client.bucket(embedding_data_bucket)
                    if markdown_content:
                        embedding_blob = embedding_bucket.blob(f"url/{query_id}_embedding.npy")
                    else:
                        embedding_blob = embedding_bucket.blob(f"url/{query_id}_embedding_embedding.npy")
                    embedding_blob.upload_from_filename(local_embedding_path)
                    st.write("Text embedding uploaded to GCS.")
                    logging.info(f"Text embedding uploaded to gs://{embedding_data_bucket}/url/{query_id}_embedding.npy")
                except Exception as e:
                    st.error(f"Failed to upload embedding to GCS: {e}")
                    logging.error(f"Failed to upload embedding to GCS: {e}", exc_info=True)
                    os.remove(local_embedding_path)
                    st.stop()

                data["embedding_path"] = f"gs://{embedding_data_bucket}/url/{query_id}_embedding.npy"

                try:
                    os.remove(local_embedding_path)
                    logging.info(f"Local embedding file {local_embedding_path} removed after upload.")
                except Exception as e:
                    logging.warning(f"Failed to remove local embedding file {local_embedding_path}: {e}")

            #     # Add text embedding to Chroma DB
            #     st.write("Adding text embedding to Chroma DB...")
            #     try:
            #         chroma_collection.add(
            #             documents=[text_to_embed],
            #             embeddings=[embedding_vector.tolist()],
            #             ids=[query_id],
            #             metadatas=[{
            #                 "raw_data_path": data["raw_data_path"],
            #                 "user_id": user_id,
            #                 "timestamp": data["timestamp"],
            #                 "modality": modality.lower()
            #             }]
            #         )
            #         st.write("Text embedding added to Chroma DB.")
            #         logging.info(f"Text embedding added to Chroma DB with ID {query_id}.")
            #     except Exception as e:
            #         st.error(f"Failed to add text embedding to Chroma DB: {e}")
            #         logging.error(f"Failed to add text embedding to Chroma DB: {e}", exc_info=True)

            # # Process images if any
            # if images:
            #     st.info("Processing images from scraped content...")
            #     image_paths = []
            #     for idx, img_url in enumerate(images):
            #         try:
            #             img_response = requests.get(img_url, timeout=10)
            #             if img_response.status_code == 200:
            #                 img_extension = img_url.split('.')[-1].split('?')[0]  # Handle query params
            #                 img_extension = img_extension.lower() if img_url else 'jpg'
            #                 img_extension = img_extension if img_extension in ['png', 'jpg', 'jpeg'] else 'jpg'
            #                 local_image_path = f"./temp/{query_id}_image_{idx}.{img_extension}"
            #                 with open(local_image_path, 'wb') as img_file:
            #                     img_file.write(img_response.content)
            #                 image_paths.append(local_image_path)
            #                 st.write(f"Downloaded image {idx+1} from scraped content.")
            #                 logging.info(f"Downloaded image {idx+1} from {img_url}.")
            #         except Exception as e:
            #             st.warning(f"Failed to download image {idx+1} from scraped content: {e}")
            #             logging.warning(f"Failed to download image {idx+1} from scraped content: {e}")

            #     # Upload images to GCS and generate embeddings
            #     if image_paths:
            #         for idx, local_image_path in enumerate(image_paths):
            #             try:
            #                 # Upload image to GCS
            #                 bucket = storage_client.bucket(raw_data_bucket)
            #                 blob = bucket.blob(f"url_images/{query_id}_image_{idx}.{local_image_path.split('.')[-1]}")
            #                 blob.upload_from_filename(local_image_path)
            #                 st.write(f"Uploaded image {idx+1} to GCS.")
            #                 logging.info(f"Uploaded image {idx+1} to gs://{raw_data_bucket}/url_images/{query_id}_image_{idx}.{local_image_path.split('.')[-1]}")
            #                 data[f"image_{idx}_raw_data_path"] = f"gs://{raw_data_bucket}/url_images/{query_id}_image_{idx}.{local_image_path.split('.')[-1]}"
            #             except Exception as e:
            #                 st.error(f"Failed to upload image {idx+1} to GCS: {e}")
            #                 logging.error(f"Failed to upload image {idx+1} to GCS: {e}", exc_info=True)
            #                 os.remove(local_image_path)
            #                 continue

            #             # Generate and store embedding for image
            #             st.info(f"Generating embedding for image {idx+1}...")
            #             try:
            #                 image = Image.open(local_image_path).convert("RGB")
            #                 inputs = clip_processor(images=image, return_tensors="pt")
            #                 with torch.no_grad():
            #                     outputs = clip_model.get_image_features(**inputs)
            #                 image_embedding_vector = outputs.detach().numpy().flatten()
            #                 image_embedding_vector /= np.linalg.norm(image_embedding_vector)
            #                 logging.info(f"Image {idx+1} embedding generated successfully.")
            #             except Exception as e:
            #                 st.error(f"Failed to generate embedding for image {idx+1}: {e}")
            #                 logging.error(f"Failed to generate embedding for image {idx+1}: {e}", exc_info=True)
            #                 os.remove(local_image_path)
            #                 continue

            #             # Save embedding as .npy locally and upload to GCS
            #             local_image_embedding_path = f"./temp/{query_id}_image_{idx}_embedding.npy"
            #             try:
            #                 np.save(local_image_embedding_path, image_embedding_vector)
            #                 embedding_bucket = storage_client.bucket(embedding_data_bucket)
            #                 embedding_blob = embedding_bucket.blob(f"url_images/{query_id}_image_{idx}_embedding.npy")
            #                 embedding_blob.upload_from_filename(local_image_embedding_path)
            #                 st.write(f"Image {idx+1} embedding uploaded to GCS.")
            #                 logging.info(f"Image {idx+1} embedding uploaded to gs://{embedding_data_bucket}/url_images/{query_id}_image_{idx}_embedding.npy")
            #             except Exception as e:
            #                 st.error(f"Failed to upload embedding for image {idx+1}: {e}")
            #                 logging.error(f"Failed to upload embedding for image {idx+1}: {e}", exc_info=True)
            #                 os.remove(local_image_embedding_path)
            #                 continue

            #             data[f"image_{idx}_embedding_path"] = f"gs://{embedding_data_bucket}/url_images/{query_id}_image_{idx}_embedding.npy"

            #             try:
            #                 os.remove(local_image_embedding_path)
            #                 logging.info(f"Local image embedding file {local_image_embedding_path} removed after upload.")
            #             except Exception as e:
            #                 logging.warning(f"Failed to remove local image embedding file {local_image_embedding_path}: {e}")

            #             # Add image embedding to Chroma DB
            #             st.write(f"Adding image {idx+1} embedding to Chroma DB...")
            #             try:
            #                 chroma_collection.add(
            #                     documents=[f"Image {idx+1} from URL {query_id}"],
            #                     embeddings=[image_embedding_vector.tolist()],
            #                     ids=[f"{query_id}_image_{idx}"],
            #                     metadatas=[{
            #                         "raw_data_path": data[f"image_{idx}_raw_data_path"],
            #                         "user_id": user_id,
            #                         "timestamp": data["timestamp"],
            #                         "modality": modality.lower()
            #                     }]
            #                 )
            #                 st.write(f"Image {idx+1} embedding added to Chroma DB.")
            #                 logging.info(f"Image {idx+1} embedding added to Chroma DB with ID {query_id}_image_{idx}.")
            #             except Exception as e:
            #                 st.error(f"Failed to add image {idx+1} embedding to Chroma DB: {e}")
            #                 logging.error(f"Failed to add image {idx+1} embedding to Chroma DB: {e}", exc_info=True)

            #             # Clean up local image file
            #             try:
            #                 os.remove(local_image_path)
            #                 logging.info(f"Local image file {local_image_path} removed after processing.")
            #             except Exception as e:
            #                 logging.warning(f"Failed to remove local image file {local_image_path}: {e}")

            # else:
            #     st.info("No images found in the scraped content.")

            # Save metadata to Firestore
            try:
                doc_ref = firestore_client.collection("queries").document(query_id)
                doc_ref.set(data)
                st.write("Metadata saved to Firestore.")
                logging.info(f"Metadata for {query_id} saved to Firestore.")
            except Exception as e:
                st.error(f"Failed to save metadata to Firestore: {e}")
                logging.error(f"Failed to save metadata to Firestore: {e}", exc_info=True)
                st.stop()

            st.success("Your URL content and embeddings have been uploaded successfully!")
            st.write("Query ID:", query_id)
            # st.write("Raw Data Path:", data["raw_data_path"])
            # st.write("Embedding Path:", data["embedding_path"])

            # Display image raw data paths and embedding paths if any
            for key in data:
                if key.startswith("image_") and key.endswith("_raw_data_path"):
                    st.write(f"{key}: {data[key]}")
                if key.startswith("image_") and key.endswith("_embedding_path"):
                    st.write(f"{key}: {data[key]}")

        
        # Implement a delay to allow Cloud Functions to process (e.g., 5 seconds)
        st.info("Waiting for Cloud Functions to process the data...")
        if modality == "Image" :
            time.sleep(15)  # Adjust the duration as needed
        else:
            time.sleep(3)
        #  # Retrieve and display Firestore metadata
        # st.header("Firestore Metadata")
        # try:
        #     doc_ref = firestore_client.collection("queries").document(query_id)
        #     doc = doc_ref.get()

        #     if doc.exists:
        #             doc_data = doc.to_dict()
        #             st.json(doc_data)
        #             logging.info(f"Query results for {query_id} retrieved successfully.")
        #     else:
        #             st.write("Document not found in Firestore.")
        #             logging.warning(f"Document {query_id} not found in Firestore.")
        # except Exception as e:
        #         st.error(f"Failed to retrieve document: {e}")
        #         logging.error(f"Failed to retrieve document {query_id}: {e}", exc_info=True)


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
                logging.info(f"Query results for {query_id} retrieved successfully.")
            else:
                st.write("Document not found in Firestore.")
                logging.warning(f"Document {query_id} not found in Firestore.")
        except Exception as e:
            st.error(f"Failed to retrieve document: {e}")
            logging.error(f"Failed to retrieve document {query_id}: {e}", exc_info=True)
