import os
import asyncio
import requests
import base64
import streamlit as st
from deepgram import Deepgram  # Correct import for SDK v2.12.0
from audio_recorder_streamlit import audio_recorder  # For recording audio in Streamlit
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
from google.cloud import aiplatform
import pickle
import soundfile as sf
import numpy as np
#from tangoflux import TangoFluxInference


# --------------------------
# GCP CONFIG (Vertex AI)
# --------------------------
PROJECT_ID = "stable-furnace-451600-r8"
REGION = "us-central1"

LLaVA_ENDPOINT_ID = "663829045758132224"
SD_ENDPOINT_ID = "326622023658766336"
MISTRAL_ENDPOINT_ID = "1636043615316738048"

API_URL = f"https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{REGION}/endpoints/{LLaVA_ENDPOINT_ID}:predict"


# --------------------------
# Vertex AI Endpoint Loaders
# --------------------------

@st.cache_resource
def get_sd_endpoint():
    aiplatform.init(project=PROJECT_ID, location=REGION)
    return aiplatform.Endpoint(endpoint_name=SD_ENDPOINT_ID)

@st.cache_resource
def get_mistral_endpoint():
    print(f"DEBUG: Using credentials from = {os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')}")
    aiplatform.init(project=PROJECT_ID, location=REGION)
    return aiplatform.Endpoint(endpoint_name=MISTRAL_ENDPOINT_ID)


# --------------------------
# Deepgram API (Speech-to-Text and Text-to-Speech)
# --------------------------
DEEPGRAM_API_KEY = "16b691e04307a2eb6f22037aa88ba0989dbee035"


# Updated to work with SDK v2.12.0
async def transcribe_audio(audio_filepath):
    try:
        # Create the Deepgram client instance
        dg_client = Deepgram(DEEPGRAM_API_KEY)

        with open(audio_filepath, 'rb') as audio_file:
            buffer_data = audio_file.read()

        # Send the transcription request
        response = await dg_client.transcription.prerecorded(
            source={"buffer": buffer_data, "mimetype": "audio/wav"},
            options={"punctuate": True}
        )

        transcription = response['results']['channels'][0]['alternatives'][0]['transcript']
        return transcription
    except Exception as e:
        st.error(f"Error during transcription: {str(e)}")
        return None

def generate_audio_from_text(text, filename="audio.mp3"):
    try:
        url = "https://api.deepgram.com/v1/speak"
        headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}", "Content-Type": "application/json"}
        payload = {"text": text}

        response = requests.post(url, headers=headers, json=payload)

        if response.status_code == 200:
            with open(filename, "wb") as f:
                f.write(response.content)
            return filename
        else:
            st.error(f"Failed to generate audio: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error generating audio: {str(e)}")
        return None


# --------------------------
# Image Captioning with LLaVA (Vertex AI)
# --------------------------

def generate_caption_from_image(image_bytes, prompt="Describe this image:"):
    if "<image>" not in prompt:
        prompt += " <image>"
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "instances": [{"image_base64": image_b64, "prompt": prompt}]
    }
    token = os.popen("gcloud auth print-access-token").read().strip()
    if not token:
        return "No access token found. Run `gcloud auth login` and try again."
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        return result["predictions"][0].get("caption", "No caption returned.")
    else:
        return f"Error {response.status_code}: {response.text}"


# --------------------------
# Image Generation (Stable Diffusion)
# --------------------------

def generate_image(prompt: str):
    endpoint = get_sd_endpoint()
    response = endpoint.predict(instances=[{"prompt": prompt}])
    base64_image = response.predictions[0].get("image_base64")
    if not base64_image:
        return None
    image_bytes = base64.b64decode(base64_image)
    return Image.open(BytesIO(image_bytes))


# --------------------------
# Text Generation (Mistral)
# --------------------------

def generate_text(prompt: str):
    endpoint = get_mistral_endpoint()
    response = endpoint.predict(instances=[{"prompt": prompt}])

    if isinstance(response.predictions, list) and len(response.predictions) > 0:
        return response.predictions[0]
    return None


# --------------------------
# PDF and URL Content Extraction
# --------------------------

def extract_text_from_pdf(pdf_file):
    try:
        reader = PdfReader(pdf_file)
        pdf_text = ""
        for page in reader.pages:
            pdf_text += page.extract_text()
        return pdf_text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None

def extract_text_from_url(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            paragraphs = soup.find_all("p")
            content = "\n".join([para.get_text() for para in paragraphs])
            return content
        else:
            st.error(f"Failed to fetch URL content, Status code: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error extracting content from URL: {str(e)}")
        return None


# --------------------------
# TangoFlux - Text to Sound Generation
# --------------------------

# Load the TangoFlux model from the saved pickle file
def load_model():
    model_path = "models/tangoflux/tangoflux_model.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

# Generate audio from the model based on the user's text input
def generate_audio(model, text_prompt, steps=5, duration=2):
    audio = model.generate(text_prompt, steps=steps, duration=duration)
    
    # If the audio is a list or other non-numpy format, convert it to numpy array
    if isinstance(audio, list):
        audio = np.array(audio)
    
    # Ensure the audio is a 1D numpy array representing the waveform
    if audio.ndim > 1:
        audio = audio.flatten()  # Flatten to 1D if necessary
    return audio


# --------------------------
# Streamlit UI
# --------------------------

st.set_page_config(page_title="Multimodal Playground", layout="centered")
st.title("Multimodal Content Generator")

# Dropdown menu for task selection
mode = st.selectbox("Choose a task:", [
    "Speech to Text", 
    "Text to Speech", 
    "Image Generation", 
    "Text Generation", 
    "Image Captioning", 
    "PDF Upload", 
    "URL Content Extraction",
    "Text to Sound Generation"
])


# --------------------------
# Task Handling
# --------------------------

# Speech to Text
if mode == "Speech to Text":
    st.subheader("Speech to Text (Upload Audio)")
    audio_file = st.file_uploader("Upload Audio File", type=["wav", "mp3", "flac"])
    if audio_file:
        st.audio(audio_file, format="audio/wav")
        if st.button("Convert to Text"):
            with st.spinner("Transcribing..."):
                audio_path = f"./{audio_file.name}"
                with open(audio_path, "wb") as f:
                    f.write(audio_file.read())
                transcription = asyncio.run(transcribe_audio(audio_path))
                if transcription:
                    st.write("Transcription:")
                    st.text_area("Transcription", transcription, height=200)
                os.remove(audio_path)

# Text to Speech
elif mode == "Text to Speech":
    st.subheader("Text to Speech (Convert Text to Audio)")
    text_input = st.text_area("Enter Text for Speech Conversion")
    if text_input:
        if st.button("Generate Speech"):
            with st.spinner("Generating Speech..."):
                audio_filename = "generated_audio.mp3"
                audio_file = generate_audio_from_text(text_input, audio_filename)
                if audio_file:
                    st.audio(audio_file, format="audio/mp3")
                    st.success("Audio generated successfully!")
                    # Provide a download button for the generated audio file
                    with open(audio_file, "rb") as f:
                        st.download_button(
                            label="Download Audio",
                            data=f,
                            file_name="generated_audio.mp3",
                            mime="audio/mp3"
                        )

# Image Generation
elif mode == "Image Generation":
    st.subheader("Stable Diffusion Image Generation")
    prompt = st.text_input("Enter your image prompt:", value="A fantasy forest with glowing mushrooms")
    if st.button("Generate Image"):
        with st.spinner("Generating image..."):
            img = generate_image(prompt)
            if img:
                st.image(img, caption="Generated Image", use_column_width=True)
            else:
                st.error("Image generation failed or no image returned.")

# Text Generation (Mistral)
elif mode == "Text Generation":
    st.subheader("Mistral Text Generation")
    prompt = st.text_input("Enter your text prompt:", value="Tell me a story about a dragon and a knight")
    if st.button("Generate Text"):
        with st.spinner("Generating text..."):
            generated_text = generate_text(prompt)
            if generated_text:
                st.write("Generated Text:")
                st.write(generated_text)
            else:
                st.error("Text generation failed or no text returned.")

# Image Captioning with LLaVA
elif mode == "Image Captioning":
    st.subheader("Image Captioning with LLaVA")
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    prompt = st.text_input("Enter caption prompt:", value="Describe this image:")
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        if st.button("Generate Caption"):
            with st.spinner("Generating caption..."):
                img_bytes = uploaded_image.read()
                caption = generate_caption_from_image(img_bytes, prompt)
                st.write("Generated Caption:")
                st.write(caption)

# TangoFlux - Text to Sound Generation
elif mode == "Text to Sound Generation":
    st.subheader("Text to Sound Generation")
    text_prompt = st.text_area("Enter a text prompt:", "Hammer slowly hitting the wooden table")
    
    if st.button("Generate Sound"):
        # Load the model
        model = load_model()
        # Generate audio from the model based on the prompt
        audio = generate_audio(model, text_prompt)
        
        # Save the generated audio to a file
        audio_path = "generated_audio.wav"
        try:
            sf.write(audio_path, audio, 44100)  # Writing the audio in 44.1kHz format
            # Display the audio player for the user to listen to the generated audio
            st.audio(audio_path, format="audio/wav")
            st.success("Audio generated successfully!")
        except Exception as e:
            st.error(f"Error saving audio: {str(e)}")

# PDF Upload
elif mode == "PDF Upload":
    st.subheader("PDF Text Extraction")
    uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_pdf:
        pdf_text = extract_text_from_pdf(uploaded_pdf)
        if pdf_text:
            st.write("Extracted Text from PDF:")
            st.text_area("PDF Content", pdf_text, height=300)

# URL Content Extraction
elif mode == "URL Content Extraction":
    st.subheader("URL Content Extraction")
    url_input = st.text_input("Enter URL")
    if url_input:
        if st.button("Extract Content"):
            with st.spinner("Extracting content from URL..."):
                url_content = extract_text_from_url(url_input)
                if url_content:
                    st.write("Extracted Content from URL:")
                    st.text_area("URL Content", url_content, height=300)

