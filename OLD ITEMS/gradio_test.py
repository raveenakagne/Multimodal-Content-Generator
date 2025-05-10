import os
import asyncio
import requests
import base64
import gradio as gr
import tempfile
import logging
import numpy as np
import soundfile as sf
import pickle
from io import BytesIO
from PIL import Image

# Import utility functions (assuming they are correctly placed or copied)
from deepgram import Deepgram
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
from google.cloud import aiplatform
# from tangoflux import TangoFluxInference # Assuming this exists if uncommented

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------
# GCP CONFIG (Vertex AI) - Copied from Streamlit
# --------------------------
# TODO: Replace with your actual Project ID if different or use environment variables
PROJECT_ID = "stable-furnace-451600-r8"
# TODO: Replace with your actual Region if different or use environment variables
REGION = "us-central1"

# TODO: Ensure these Endpoint IDs are correct for your GCP project
LLaVA_ENDPOINT_ID = "663829045758132224"
SD_ENDPOINT_ID = "326622023658766336"
MISTRAL_ENDPOINT_ID = "1636043615316738048"

# LLaVA API URL (used for manual REST call)
LLAVA_API_URL = f"https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{REGION}/endpoints/{LLaVA_ENDPOINT_ID}:predict"

# --------------------------
# Deepgram API Key - Copied from Streamlit
# --------------------------
# TODO: It's better practice to load secrets from environment variables or a secure vault
DEEPGRAM_API_KEY = "16b691e04307a2eb6f22037aa88ba0989dbee035" # Use creds.DEEPGRAM_API_KEY if defined

# ----------------------------------
# Vertex AI Client Initialization
# ----------------------------------
# Initialize globally once when the script starts
try:
    aiplatform.init(project=PROJECT_ID, location=REGION)
    mistral_endpoint = aiplatform.Endpoint(endpoint_name=MISTRAL_ENDPOINT_ID)
    sd_endpoint = aiplatform.Endpoint(endpoint_name=SD_ENDPOINT_ID)
    logger.info("Vertex AI Endpoints initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize Vertex AI Endpoints: {e}")
    # Set endpoints to None so functions can check and fail gracefully
    mistral_endpoint = None
    sd_endpoint = None
    # Optionally, raise the exception or exit if Vertex AI is critical
    # raise e

# ----------------------------------
# Backend Processing Functions
# (Adapted from Streamlit code)
# ----------------------------------

# --- Deepgram Functions ---
async def transcribe_audio_async(audio_filepath):
    """Async function for Deepgram transcription."""
    if not DEEPGRAM_API_KEY:
        return "Error: Deepgram API Key not configured."
    try:
        dg_client = Deepgram(DEEPGRAM_API_KEY)
        with open(audio_filepath, 'rb') as audio_file:
            buffer_data = audio_file.read()

        source = {"buffer": buffer_data, "mimetype": "audio/wav"} # Adjust mimetype if needed based on input component
        options = {"punctuate": True}
        response = await dg_client.transcription.prerecorded(source, options)
        transcription = response['results']['channels'][0]['alternatives'][0]['transcript']
        return transcription
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
        return f"Error during transcription: {str(e)}"

def handle_speech_to_text_sync(audio_filepath):
    """Sync wrapper for Gradio."""
    if audio_filepath is None:
        return "Error: No audio file provided."
    try:
        # Gradio might provide formats other than wav, consider conversion if needed
        # For simplicity, assume input is compatible or handle conversion
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(transcribe_audio_async(audio_filepath))
        loop.close()
        return result
    except Exception as e:
        logger.error(f"Sync STT Handler Error: {e}")
        return f"Error: {e}"


def generate_audio_from_text(text):
    """Generates audio using Deepgram TTS and saves to a temporary file."""
    if not DEEPGRAM_API_KEY:
        return "Error: Deepgram API Key not configured.", None
    if not text:
        return "Error: No text provided.", None
    try:
        url = "https://api.deepgram.com/v1/speak"
        headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}", "Content-Type": "application/json"}
        payload = {"text": text}
        response = requests.post(url, headers=headers, json=payload)

        if response.status_code == 200:
            # Save to a temporary file for Gradio
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
                temp_audio_file.write(response.content)
                temp_audio_path = temp_audio_file.name
            logger.info(f"Generated audio saved to {temp_audio_path}")
            # Return the path for gr.Audio and gr.DownloadButton
            return temp_audio_path, temp_audio_path # Path for audio player, Path for download
        else:
            logger.error(f"Failed to generate audio: {response.status_code} - {response.text}")
            return f"Error: Failed to generate audio ({response.status_code})", None
    except Exception as e:
        logger.error(f"Error generating audio: {str(e)}")
        return f"Error generating audio: {str(e)}", None

# --- Vertex AI Functions ---

def generate_caption_vertex(image_input, prompt="Describe this image:"):
    """LLaVA image captioning via Vertex AI REST API."""
    if image_input is None:
        return "Error: No image provided."

    # Gradio gr.Image(type="filepath") provides a path
    try:
        with open(image_input, "rb") as img_file:
            image_bytes = img_file.read()
    except Exception as e:
        logger.error(f"Error reading image file: {e}")
        return f"Error reading image file: {e}"

    if "<image>" not in prompt:
        prompt += " <image>"
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {"instances": [{"image_base64": image_b64, "prompt": prompt}]}

    try:
        # Get token dynamically - requires gcloud CLI to be installed and configured
        token = os.popen("gcloud auth print-access-token").read().strip()
        if not token:
            return "Error: Could not get gcloud access token. Please run 'gcloud auth login' and 'gcloud auth application-default login'."

        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        response = requests.post(LLAVA_API_URL, headers=headers, json=payload)

        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        result = response.json()
        # Check if predictions array exists and is not empty
        if "predictions" in result and isinstance(result["predictions"], list) and len(result["predictions"]) > 0:
             # Safely get the caption
             caption = result["predictions"][0].get("caption", "No caption returned in prediction.")
             return caption
        else:
            logger.error(f"Unexpected response format from LLaVA endpoint: {result}")
            return "Error: Unexpected response format from captioning model."

    except requests.exceptions.RequestException as e:
        logger.error(f"LLaVA API Request Error: {e}")
        # Attempt to get more detailed error from response if available
        error_detail = ""
        try:
            error_detail = e.response.text
        except:
            pass # Ignore if we can't get response body
        return f"Error calling captioning API: {e}. Details: {error_detail}"
    except Exception as e:
        logger.error(f"Captioning error: {str(e)}")
        return f"Captioning error: {str(e)}"


def generate_image_vertex(prompt: str):
    """Generates image using Vertex AI Stable Diffusion Endpoint."""
    if sd_endpoint is None:
         return None, "Error: Stable Diffusion Endpoint not initialized."
    if not prompt:
        return None, "Error: Please enter a prompt."

    try:
        response = sd_endpoint.predict(instances=[{"prompt": prompt}])
        # Check response structure (this might vary based on exact model deployment)
        if not response.predictions or not isinstance(response.predictions, list) or len(response.predictions) == 0:
            logger.error(f"Unexpected SD response format: {response}")
            return None, "Error: Received unexpected response format from image generation model."

        # Safely access the prediction dictionary and the image data
        prediction = response.predictions[0]
        if not isinstance(prediction, dict) or "image_base64" not in prediction:
             logger.error(f"Prediction dictionary missing 'image_base64': {prediction}")
             return None, "Error: Prediction data missing required image information."

        base64_image = prediction["image_base64"]
        if not base64_image:
            return None, "Error: Image generation failed or no image returned."

        image_bytes = base64.b64decode(base64_image)
        img = Image.open(BytesIO(image_bytes))
        return img, None # Return PIL image and no error
    except Exception as e:
        logger.error(f"Image generation error: {str(e)}")
        # Check for specific Google Cloud API errors if needed
        return None, f"Image generation error: {str(e)}" # Return None image and error message


def generate_text_vertex(prompt: str):
    """Generates text using Vertex AI Mistral Endpoint."""
    if mistral_endpoint is None:
        return "Error: Mistral Endpoint not initialized."
    if not prompt:
        return "Error: Please enter a prompt."

    try:
        # Construct the payload according to Mistral's expected format
        # This might need adjustment based on the specific Mistral model deployment
        # Common format is a list of instances, each with a 'prompt' field
        instances = [{"prompt": prompt}]
        response = mistral_endpoint.predict(instances=instances)

        # Check response structure
        if not response.predictions or not isinstance(response.predictions, list) or len(response.predictions) == 0:
            logger.error(f"Unexpected Mistral response format: {response}")
            return "Error: Received unexpected response format from text generation model."

        # Extract the generated text (adapt based on actual response structure)
        # Often it's the first element of the predictions list
        generated_text = response.predictions[0]
        # Sometimes the prediction itself is a dict, e.g., {'generated_text': '...'}
        if isinstance(generated_text, dict):
             # Adjust key if necessary, e.g., 'text', 'output', 'content'
             generated_text = generated_text.get('generated_text', generated_text.get('content', str(generated_text)))

        return str(generated_text) # Ensure it's a string
    except Exception as e:
        logger.error(f"Text generation error: {str(e)}")
        return f"Text generation error: {str(e)}"

# --- PDF and URL Functions ---
def extract_text_from_pdf(pdf_file_obj):
    """Extracts text from PDF file object provided by Gradio."""
    if pdf_file_obj is None:
        return "Error: No PDF file provided."
    try:
        # pdf_file_obj.name contains the path to the temporary file
        reader = PdfReader(pdf_file_obj.name)
        pdf_text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text: # Add text only if extraction was successful
                 pdf_text += page_text + "\n"
        return pdf_text if pdf_text else "No text could be extracted from this PDF."
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        return f"Error extracting text from PDF: {str(e)}"

def extract_text_from_url(url):
    """Extracts paragraph text from a URL."""
    if not url:
        return "Error: No URL provided."
    try:
        # Add http:// if missing
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url

        response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'}) # Add user-agent
        response.raise_for_status() # Check for HTTP errors

        # Check content type
        if 'text/html' not in response.headers.get('Content-Type', ''):
            return f"Error: URL content type is not HTML ({response.headers.get('Content-Type', 'N/A')})"

        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        content = "\n".join([para.get_text().strip() for para in paragraphs if para.get_text().strip()])
        return content if content else "No paragraph text found at the URL."
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch URL content: {e}")
        return f"Error fetching URL: {e}"
    except Exception as e:
        logger.error(f"Error extracting content from URL: {str(e)}")
        return f"Error extracting content from URL: {str(e)}"

# --- TangoFlux Functions (Placeholder - uncomment and ensure library/model exists) ---
# Assuming 'tangoflux' library and model file exist at the specified path
TANGOFLUX_MODEL_PATH = "models/tangoflux/tangoflux_model.pkl"

def load_tangoflux_model():
    """Loads the TangoFlux model."""
    if not os.path.exists(TANGOFLUX_MODEL_PATH):
        logger.warning(f"TangoFlux model not found at {TANGOFLUX_MODEL_PATH}. Text-to-Sound disabled.")
        return None
    try:
        with open(TANGOFLUX_MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        logger.info("TangoFlux model loaded.")
        return model
    except Exception as e:
        logger.error(f"Error loading TangoFlux model: {e}")
        return None

# Load model globally if desired (can take time) or load within handler
# tango_model = load_tangoflux_model() # Optional global load

def generate_sound_tangoflux(text_prompt):
    """Generates sound using TangoFlux model."""
    # Load model here if not loaded globally
    tango_model = load_tangoflux_model()
    if tango_model is None:
        return None, "Error: TangoFlux model could not be loaded."
    if not text_prompt:
         return None, "Error: Please enter a sound prompt."

    try:
        # Adjust steps/duration as needed
        # Assuming model.generate returns a numpy array (waveform) and sample rate
        # Modify based on actual TangoFlux library API
        # audio_waveform, sample_rate = tango_model.generate(text_prompt, steps=5, duration=2) # Example
        audio_waveform = np.random.rand(44100 * 2) # Placeholder: Replace with actual model call
        sample_rate = 44100 # Placeholder: Replace with actual model sample rate

        if not isinstance(audio_waveform, np.ndarray):
            return None, "Error: Sound generation did not return a valid audio format."

        # Save the generated audio to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
            sf.write(temp_audio_file.name, audio_waveform, sample_rate)
            temp_audio_path = temp_audio_file.name
        logger.info(f"Generated sound saved to {temp_audio_path}")
        return temp_audio_path, None # Return path and no error

    except Exception as e:
        logger.error(f"Error generating sound: {str(e)}")
        return None, f"Error generating sound: {str(e)}"


# --------------------------
# Gradio UI Definition
# --------------------------
with gr.Blocks(title="Multimodal Playground", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Multimodal Content Generator")
    gr.Markdown("Select a task to generate content.")

    mode = gr.Dropdown(
        label="Choose Task",
        choices=[
            "Speech to Text",
            "Text to Speech",
            "Image Generation",
            "Text Generation",
            "Image Captioning",
            "PDF Upload",
            "URL Content Extraction",
            "Text to Sound Generation"
        ],
        value="Text Generation", # Default task
        interactive=True
    )

    # --- Task Interfaces ---

    # Speech to Text
    with gr.Column(visible=False) as speech_col:
        with gr.Row():
            audio_input_stt = gr.Audio(label="Upload Audio File", type="filepath", sources=["upload", "microphone"])
        with gr.Row():
            transcribe_btn = gr.Button("Transcribe Audio")
        with gr.Row():
            transcription_output = gr.Textbox(label="Transcription", lines=5, interactive=False)

    # Text to Speech
    with gr.Column(visible=False) as tts_col:
        with gr.Row():
            text_input_tts = gr.Textbox(label="Enter Text for Speech", lines=3)
        with gr.Row():
            tts_btn = gr.Button("Generate Speech")
        with gr.Row():
            # Output audio player
            audio_output_tts = gr.Audio(label="Generated Audio", type="filepath", interactive=False)
        with gr.Row():
             # Output for download button (uses the same path)
             tts_download_path = gr.State(None) # Store path for download
             tts_download_btn = gr.DownloadButton(label="Download Audio", visible=False)


    # Image Generation
    with gr.Column(visible=False) as img_gen_col:
        with gr.Row():
            img_prompt_gen = gr.Textbox(label="Enter Image Prompt", value="A photo of an astronaut riding a horse on the moon")
        with gr.Row():
            generate_img_btn = gr.Button("Generate Image")
        with gr.Row():
            image_output_gen = gr.Image(label="Generated Image", type="pil", interactive=False) # Output as PIL
            img_gen_error_output = gr.Textbox(label="Error", visible=False, interactive=False)


    # Text Generation
    with gr.Column(visible=False) as text_gen_col:
        with gr.Row():
            text_prompt_gen = gr.Textbox(label="Enter Text Prompt", value="Write a short poem about a rainy day.")
        with gr.Row():
            generate_text_btn = gr.Button("Generate Text")
        with gr.Row():
            text_output_gen = gr.Textbox(label="Generated Text", lines=8, interactive=False)

    # Image Captioning
    with gr.Column(visible=False) as img_cap_col:
        with gr.Row():
             image_input_cap = gr.Image(label="Upload Image", type="filepath", sources=["upload"]) # Needs filepath for reading
        with gr.Row():
             caption_prompt_cap = gr.Textbox(label="Optional: Modify Caption Prompt", value="Describe this image in detail:")
        with gr.Row():
             caption_btn = gr.Button("Generate Caption")
        with gr.Row():
             caption_output_cap = gr.Textbox(label="Generated Caption", lines=4, interactive=False)

    # PDF Upload
    with gr.Column(visible=False) as pdf_col:
        with gr.Row():
            pdf_input_ext = gr.File(label="Upload PDF File", file_types=[".pdf"])
        with gr.Row():
            pdf_output_ext = gr.Textbox(label="Extracted Text", lines=10, interactive=False)

    # URL Content Extraction
    with gr.Column(visible=False) as url_col:
        with gr.Row():
            url_input_ext = gr.Textbox(label="Enter URL (e.g., https://www.example.com)")
        with gr.Row():
            extract_btn = gr.Button("Extract Content from URL")
        with gr.Row():
            url_output_ext = gr.Textbox(label="Extracted Content", lines=10, interactive=False)

    # Text to Sound Generation
    with gr.Column(visible=False) as sound_col:
        with gr.Row():
            sound_prompt_gen = gr.Textbox(label="Enter Sound Description", value="Ocean waves crashing on a beach")
        with gr.Row():
            sound_btn = gr.Button("Generate Sound")
        with gr.Row():
            sound_output_gen = gr.Audio(label="Generated Sound", type="filepath", interactive=False)
            sound_gen_error_output = gr.Textbox(label="Error", visible=False, interactive=False)


    # --- Dynamic Visibility Logic ---
    task_map = {
        "Speech to Text": speech_col,
        "Text to Speech": tts_col,
        "Image Generation": img_gen_col,
        "Text Generation": text_gen_col,
        "Image Captioning": img_cap_col,
        "PDF Upload": pdf_col,
        "URL Content Extraction": url_col,
        "Text to Sound Generation": sound_col
    }

    def update_visibility(selected_mode):
        updates = {}
        for task_name, component in task_map.items():
            updates[component] = gr.Column(visible=(task_name == selected_mode))
        return updates

    mode.change(
        fn=update_visibility,
        inputs=mode,
        outputs=list(task_map.values())
    )

    # --- Event Handlers ---

    # Speech to Text
    transcribe_btn.click(
        fn=handle_speech_to_text_sync,
        inputs=[audio_input_stt],
        outputs=[transcription_output]
    )

    # Text to Speech
    def tts_wrapper(text):
        audio_path, download_uri = generate_audio_from_text(text)
        # Show download button only if generation succeeded
        show_download = download_uri is not None
        return {
            audio_output_tts: gr.Audio(value=audio_path, visible=audio_path is not None),
            tts_download_path: download_uri, # Store path for download button
            tts_download_btn: gr.DownloadButton(value=download_uri, visible=show_download) # Update download button
        }
    tts_btn.click(
        fn=tts_wrapper,
        inputs=[text_input_tts],
        outputs=[audio_output_tts, tts_download_path, tts_download_btn] # Update player, path state, and download button
    )

    # Image Generation
    def img_gen_wrapper(prompt):
        img, error = generate_image_vertex(prompt)
        return {
            image_output_gen: gr.Image(value=img, visible=img is not None),
            img_gen_error_output: gr.Textbox(value=error, visible=error is not None)
        }
    generate_img_btn.click(
        fn=img_gen_wrapper,
        inputs=[img_prompt_gen],
        outputs=[image_output_gen, img_gen_error_output]
    )

    # Text Generation
    generate_text_btn.click(
        fn=generate_text_vertex,
        inputs=[text_prompt_gen],
        outputs=[text_output_gen]
    )

    # Image Captioning
    caption_btn.click(
        fn=generate_caption_vertex,
        inputs=[image_input_cap, caption_prompt_cap],
        outputs=[caption_output_cap]
    )

    # PDF Upload
    pdf_input_ext.upload( # Use upload for immediate processing
        fn=extract_text_from_pdf,
        inputs=[pdf_input_ext],
        outputs=[pdf_output_ext]
    )

    # URL Content Extraction
    extract_btn.click(
        fn=extract_text_from_url,
        inputs=[url_input_ext],
        outputs=[url_output_ext]
    )

    # Text to Sound Generation
    def sound_gen_wrapper(prompt):
         path, error = generate_sound_tangoflux(prompt)
         return {
             sound_output_gen: gr.Audio(value=path, visible=path is not None),
             sound_gen_error_output: gr.Textbox(value=error, visible=error is not None)
         }
    sound_btn.click(
        fn=sound_gen_wrapper,
        inputs=[sound_prompt_gen],
        outputs=[sound_output_gen, sound_gen_error_output]
    )

# --- Launch the Gradio App ---
if __name__ == "__main__":
    logger.info("Starting Gradio application...")
    # Set share=True if you need a public link (be careful with security/API keys)
    demo.queue().launch(server_port=7860, show_error=True, share=False)
    logger.info("Gradio application stopped.")