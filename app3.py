import os
import asyncio
import streamlit as st
from transformers import LlavaForConditionalGeneration, LlavaProcessor, AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline
from audio_recorder_streamlit import audio_recorder
# Update your imports (remove old Deepgram import)
# from deepgram import DeepgramClient, SpeakOptions, PrerecordedOptions
from deepgram import Deepgram
import requests
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
 
 
# Hugging Face login (replace with your token)
from huggingface_hub import login
login(token="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")
 
# Deepgram API Key (replace with your actual key)
DEEPGRAM_API_KEY = "16b691e04307a2eb6f22037aa88ba0989dbee035"
 
# Disable Streamlit file watcher
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
 
# Ensure the event loop is set up correctly
try:
    loop = asyncio.get_event_loop()
except RuntimeError as e:
    if "no current event loop" in str(e):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    else:
        raise
 
# Cache the model loading to avoid reloading on every interaction
@st.cache_resource
def load_llava_model():
    model_name = "llava-hf/llava-1.5-7b-hf"
    processor = LlavaProcessor.from_pretrained(model_name)
    model = LlavaForConditionalGeneration.from_pretrained(model_name)
    return processor, model
 
@st.cache_resource
def load_stable_diffusion_model():
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe
 
@st.cache_resource
def load_mistral_model():
    model_name = "mistralai/Mistral-7B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model
 
# Load models
processor, llava_model = load_llava_model()
pipe = load_stable_diffusion_model()
mistral_tokenizer, mistral_model = load_mistral_model()
 
# Function to transcribe audio using Deepgram
async def transcribe_audio(audio_filepath):
    try:
        # Initialize the Deepgram client
        dg_client = Deepgram(DEEPGRAM_API_KEY)
 
        # Read the audio file
        with open(audio_filepath, 'rb') as audio_file:
            buffer_data = audio_file.read()
 
        # Transcribe the audio file (asynchronous)
        response = await dg_client.transcription.prerecorded(
            source={"buffer": buffer_data, "mimetype": "audio/wav"},
            options={"model": "nova-2", "punctuate": True}
        )
 
        # Extract the transcription from the response
        transcription = response["results"]["channels"][0]["alternatives"][0]["transcript"]
        return transcription
 
    except Exception as e:
        raise Exception(f"Error during transcription: {str(e)}")
 
# Function to generate caption from image using LLaVA
def generate_caption_from_image(image_file):
    try:
        image = Image.open(image_file).convert("RGB")
        prompt = "Describe this image: <image>"
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(llava_model.device) for k, v in inputs.items()}
        generated_ids = llava_model.generate(**inputs, max_new_tokens=100)
        caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return caption
    except Exception as e:
        st.error(f"Error generating caption: {str(e)}")
        return None
 
# Function to generate image from text using Stable Diffusion (only if requested)
def generate_image_from_text(text):
    if "generate an image" in text.lower() or "create an image" in text.lower():
        image = pipe(text).images[0]
        return image
    return None
 
# Function to generate text response using Mistral with improved decoding strategies
def generate_text_response(prompt):
    try:
        inputs = mistral_tokenizer(prompt, return_tensors="pt")
        outputs = mistral_model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            top_k=50,
            temperature=0.7,
            eos_token_id=mistral_tokenizer.eos_token_id
        )
        response = mistral_tokenizer.decode(outputs[0], skip_special_tokens=True)
 
        # Post-process to ensure proper sentence completion
        if not response.endswith(('.', '!', '?')):
            response += "..."
 
        return response.strip()
    except Exception as e:
        st.error(f"Error generating text response: {str(e)}")
        return None
 
# Function to generate audio from text using Deepgram
# def generate_audio_from_text(text, filename="audio.mp3"):
#     try:
#         deepgram = Deepgram("16b691e04307a2eb6f22037aa88ba0989dbee035")
 
#         options = SpeakOptions(
#             model="aura-asteria-en",
#         )
 
#         response = deepgram.speak.v("1").save(filename, {"text": text}, options)
#         response.json()
#         return filename
#     except Exception as e:
#         st.error(f"Error generating audio: {str(e)}")
#         return None
from deepgram import Deepgram
 
# Function to generate audio from text using Deepgram (v2)
import requests
 
# # Function to generate audio from text using Deepgram TTS API
# def generate_audio_from_text(text, filename="audio.mp3"):
#     try:
#         # Deepgram TTS API endpoint
#         url = "https://api.deepgram.com/v1/speak"
        
#         # Request headers
#         headers = {
#             "Authorization": f"Token {DEEPGRAM_API_KEY}",
#             "Content-Type": "application/json",
#         }
        
#         # Request payload
#         payload = {
#             "text": text,
#             "model": "aura-asteria-en",
#         }
        
#         # Send POST request to Deepgram TTS API
#         response = requests.post(url, headers=headers, json=payload)
        
#         # Check if the request was successful
#         if response.status_code == 200:
#             # Save the audio file
#             with open(filename, "wb") as f:
#                 f.write(response.content)
#             return filename
#         else:
#             st.error(f"Failed to generate audio: {response.text}")
#             return None
#     except Exception as e:
#         st.error(f"Error generating audio: {str(e)}")
#         return None
 
def generate_audio_from_text(text, filename="audio.mp3"):
    try:
        # Deepgram TTS API endpoint
        url = "https://api.deepgram.com/v1/speak"
        
        # Request headers
        headers = {
            "Authorization": f"Token {DEEPGRAM_API_KEY}",
            "Content-Type": "application/json",
        }
        
        # Request payload - only include the text field
        payload = {
            "text": text
        }
        
        # Send POST request to Deepgram TTS API
        response = requests.post(url, headers=headers, json=payload)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Save the audio file
            with open(filename, "wb") as f:
                f.write(response.content)
            return filename
        else:
            st.error(f"Failed to generate audio: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error generating audio: {str(e)}")
        return None
  
# Streamlit UI
st.title("Multimodal Content Generation")
 
# User ID (optional)
user_id = st.text_input("User ID (optional)")
 
# Select Input Modality
modality = st.selectbox(
    "Select Input Modality",
    options=["Text", "Image", "Audio", "PDF", "URL", "Text-to-Audio"],
    index=0  # Default to Text
)
 
# Input based on selected modality
if modality == "Text":
    text_query = st.text_area("Enter your text query")
elif modality == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
elif modality == "Audio":
    audio_option = st.radio("Choose audio input method:", ("Upload", "Record"))
    if audio_option == "Upload":
        uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"])
    else:
        audio_bytes = audio_recorder()
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
elif modality == "PDF":
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
elif modality == "URL":
    url = st.text_input("Enter URL")
elif modality == "Text-to-Audio":
    text_for_audio = st.text_area("Enter text to convert to audio")
 
# Submit Button Logic for All Modalities
if st.button("Submit"):
    if modality == "Text" and text_query:
        st.success(f"Text query submitted: {text_query}")
        with st.spinner("Generating text response..."):
            text_response = generate_text_response(text_query)
        if text_response:
            st.write("Generated Text Response:")
            st.write(text_response)
 
        # Generate an image only if explicitly requested in the prompt
        generated_image = generate_image_from_text(text_query)
        if generated_image:
            st.image(generated_image, caption="Generated Image", use_container_width=True)
 
    elif modality == "Image" and uploaded_file is not None:
        st.success("Image uploaded successfully!")
        with st.spinner("Generating caption..."):
            caption = generate_caption_from_image(uploaded_file)
        if caption:
            st.write(f"Generated Caption: {caption}")
            with st.spinner("Generating image from caption..."):
                generated_image = generate_image_from_text(caption)
            if generated_image:
                st.image(generated_image, caption="Generated Image", use_container_width=True)
 
    elif modality == "Audio":
        if audio_option == "Upload" and uploaded_file is not None:
            st.success("Audio file uploaded successfully!")
        
            # Save uploaded file temporarily for processing.
            with open(uploaded_file.name, 'wb') as f:
                f.write(uploaded_file.read())
                st.write(f"File saved at: {uploaded_file.name}, size: {uploaded_file.size} bytes.")
 
            with st.spinner("Transcribing audio..."):
                try:
                    # Run transcription asynchronously and get the result
                    transcript = asyncio.run(transcribe_audio(uploaded_file.name))
                    if transcript.strip():
                        st.write("Transcript:")
                        st.text_area(label="Transcript", value=transcript, height=200)
                    else:
                        st.warning("No transcript found. Please ensure the audio contains clear speech.")
                except Exception as e:
                    st.error(f"Error transcribing audio: {str(e)}")
 
            os.remove(uploaded_file.name)  # Clean up temporary file after processing.
 
        elif audio_option == "Record" and audio_bytes is not None:
            st.success("Audio recorded successfully!")
        
            # Save recorded audio temporarily for processing.
            with open("temp_audio.wav", 'wb') as f:
                f.write(audio_bytes)
        
            with st.spinner("Transcribing audio..."):
                try:
                    # Run transcription asynchronously and get the result
                    transcript = asyncio.run(transcribe_audio("temp_audio.wav"))
                    if transcript.strip():
                        st.write("Transcript:")
                        st.text_area(label="Transcript", value=transcript, height=200)
                    else:
                        st.warning("No transcript found. Please ensure the audio contains clear speech.")
                except Exception as e:
                    st.error(f"Error transcribing audio: {str(e)}")
 
            os.remove("temp_audio.wav")  # Clean up temporary file after processing.
 
    elif modality == "PDF" and uploaded_file is not None:
        st.success("PDF file uploaded successfully!")
        # Process PDF file
        try:
            with open(uploaded_file.name, 'wb') as f:
                f.write(uploaded_file.read())
            
            reader = PdfReader(uploaded_file.name)
            pdf_text = ""
            
            # Extract text from each page
            for page in reader.pages:
                pdf_text += page.extract_text()
            
            os.remove(uploaded_file.name)  # Clean up temporary file after processing
            
            # Display extracted text
            if pdf_text.strip():
                st.write("Extracted Text from PDF:")
                st.text_area(label="PDF Content", value=pdf_text, height=300)
            else:
                st.warning("No text could be extracted from the PDF.")
        
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
 
    elif modality == "URL" and url:
        st.success(f"Processing URL: {url}")
        try:
            # Step 1: Check if the URL is accessible
            st.info("Checking URL accessibility...")
            try:
                response = requests.get(url, timeout=10)
                if response.status_code != 200:
                    st.error(f"URL returned status code {response.status_code}. Please check the link.")
                    st.stop()
            except Exception as e:
                st.error(f"Failed to reach the URL: {e}")
                st.stop()
 
            # Step 2: Extract content from the URL
            st.info("Fetching content from the URL...")
            from bs4 import BeautifulSoup
 
            # Parse HTML content
            soup = BeautifulSoup(response.text, "html.parser")
 
            # Extract main content (e.g., <p> tags)
            main_content = ""
            for paragraph in soup.find_all("p"):
                main_content += paragraph.get_text() + "\n\n"
 
            # Step 3: Display extracted content
            if main_content.strip():
                st.write("Extracted Content:")
                st.text_area(label="Main Content", value=main_content, height=300)
            else:
                st.warning("No readable text found on the page.")
 
        except Exception as e:
            st.error(f"An error occurred while processing the URL: {e}")
 
    elif modality == "Text-to-Audio" and text_for_audio:
        st.success("Text submitted for audio generation!")
        with st.spinner("Generating audio..."):
            audio_file = generate_audio_from_text(text_for_audio)
            if audio_file:
                st.audio(audio_file, format="audio/mp3")
            else:
                st.error("Failed to generate audio.")