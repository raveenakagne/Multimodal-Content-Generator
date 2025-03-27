import os
import asyncio
import gradio as gr
import torch
from transformers import (
    LlavaForConditionalGeneration, 
    LlavaProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from diffusers import StableDiffusionPipeline
from deepgram import Deepgram
import requests
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
import tempfile
import logging
import creds
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Hardware Configuration ---
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"

# --- Model Quantization Config ---
bnb_config = None
if DEVICE == "cpu":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

def load_model(model_name, model_class, **kwargs):
    """Safe loader for models without triggering offload-related conflicts"""
    try:
        if DEVICE == "cpu":
            # Avoid device_map if using CPU
            model = model_class.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                **kwargs
            )
            model.to("cpu")  # Safe manual move
        elif DEVICE == "mps":
            # MPS supports float32 only
            model = model_class.from_pretrained(
                model_name,
                device_map={"": DEVICE},
                torch_dtype=torch.float32,
                **kwargs
            )
        else:  # GPU
            model = model_class.from_pretrained(
                model_name,
                device_map="auto",  # Offload to GPU only
                torch_dtype=torch.float16,
                **kwargs
            )
        return model.eval()
    except Exception as e:
        logger.error(f"Error loading {model_name}: {str(e)}")
        raise

# Load models on demand
class ModelManager:
    # def __init__(self):
    #     self.models = {}
    #     # Initialize Deepgram client for v2.12.0
    #     from deepgram import DeepgramClient
    #     self.deepgram = DeepgramClient("16b691e04307a2eb6f22037aa88ba0989dbee035")
    def __init__(self):
        self.models = {}
        # For Deepgram v2.12.0
        from deepgram import Deepgram
        self.deepgram = Deepgram("16b691e04307a2eb6f22037aa88ba0989dbee035")
  
    def get_llava(self):
        if "llava" not in self.models:
            processor = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
            model = load_model(
                "llava-hf/llava-1.5-7b-hf",
                LlavaForConditionalGeneration
            )
            self.models["llava"] = (processor, model)
        return self.models["llava"]
    
    def get_sd(self):
        if "sd" not in self.models:
            self.models["sd"] = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
            ).to(DEVICE)
        return self.models["sd"]
    
    def get_mistral(self):
        if "mistral" not in self.models:
            tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
            model = load_model(
                "mistralai/Mistral-7B-v0.1",
                AutoModelForCausalLM
            )
            self.models["mistral"] = (tokenizer, model)
        return self.models["mistral"]
    
    def cleanup(self):
        for name in list(self.models.keys()):
            del self.models[name]
            torch.cuda.empty_cache() if DEVICE == "cuda" else None
            logger.info(f"Unloaded {name} model")

model_manager = ModelManager()

# --- Processing Functions ---
async def safe_transcribe(audio_path):
    """Memory-safe audio transcription"""
    try:
        with open(audio_path, "rb") as f:
            source = {"buffer": f.read(), "mimetype": 'audio/wav'}
        
        response = await model_manager.deepgram.transcription.prerecorded(
            source,
            {"punctuate": True, "model": "nova-2"}
        )
        return response['results']['channels'][0]['alternatives'][0]['transcript']
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        return None
    finally:
        del source
        torch.cuda.empty_cache() if DEVICE == "cuda" else None

def process_image(image_path):
    """Image processing with automatic cleanup"""
    try:
        processor, model = model_manager.get_llava()
        image = Image.open(image_path).convert("RGB")
        
        inputs = processor(
            text="Describe this image: <image>", 
            images=image, 
            return_tensors="pt"
        ).to(DEVICE)
        
        generated_ids = model.generate(**inputs, max_new_tokens=100)
        caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return caption
    finally:
        del inputs, image
        torch.cuda.empty_cache() if DEVICE == "cuda" else None

def process_pdf(file_path):
    try:
        with open(file_path, "rb") as f:
            reader = PdfReader(f)
            return "\n".join([page.extract_text() for page in reader.pages])
    except Exception as e:
        return f"PDF processing error: {str(e)}"

def process_url(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        return "\n\n".join([p.get_text() for p in soup.find_all("p")])
    except Exception as e:
        return f"URL processing error: {str(e)}"

async def process_input(modality, text_query, image_path, audio_path, pdf_file, url_input, tts_text):
    """Main processing function handling all modalities"""
    outputs = {"text": "", "image": None, "audio": None}
    
    try:
        if modality == "Text" and text_query:
            # Text generation
            tokenizer, model = model_manager.get_mistral()
            inputs = tokenizer(text_query, return_tensors="pt").to(DEVICE)
            generated = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                top_k=50,
                temperature=0.7
            )
            outputs["text"] = tokenizer.decode(generated[0], skip_special_tokens=True)
            
            if "generate an image" in text_query.lower():
                pipe = model_manager.get_sd()
                outputs["image"] = pipe(text_query).images[0]

        elif modality == "Image" and image_path:
            caption = process_image(image_path)
            outputs["text"] = caption
            pipe = model_manager.get_sd()
            outputs["image"] = pipe(caption).images[0]

        elif modality == "Audio" and audio_path:
            transcript = await safe_transcribe(audio_path)
            outputs["text"] = transcript

        elif modality == "PDF" and pdf_file:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(pdf_file.read())
                outputs["text"] = process_pdf(tmp.name)
            os.unlink(tmp.name)

        elif modality == "URL" and url_input:
            outputs["text"] = process_url(url_input)

        elif modality == "Text-to-Audio" and tts_text:
            response = requests.post(
                "https://api.deepgram.com/v1/speak",
                headers={"Authorization": f"Token 16b691e04307a2eb6f22037aa88ba0989dbee035"},
                json={"text": tts_text}
            )
            if response.status_code == 200:
                outputs["audio"] = ("audio.mp3", response.content)

    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        outputs["text"] = f"Error: {str(e)}"
    
    return outputs["text"], outputs["image"], outputs["audio"]


# --- Gradio Interface ---
with gr.Blocks(title="Multimodal Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 Multimodal Content Generator")
    
    with gr.Row():
        modality = gr.Dropdown(
            label="Input Type",
            choices=["Text", "Image", "Audio", "PDF", "URL", "Text-to-Audio"],
            value="Text"
        )
    
    with gr.Tabs():
        with gr.Tab("Text"):
            text_input = gr.Textbox(label="Input Text", lines=3)
        
        with gr.Tab("Image"):
            image_input = gr.Image(type="filepath")
        
        with gr.Tab("Audio"):
            audio_input = gr.Audio(type="filepath")
        
        with gr.Tab("PDF"):
            pdf_input = gr.File(file_types=[".pdf"])
        
        with gr.Tab("URL"):
            url_input = gr.Textbox(label="Website URL")
        
        with gr.Tab("Text-to-Audio"):
            tts_input = gr.Textbox(label="Text to Convert", lines=3)
    
    submit_btn = gr.Button("Generate", variant="primary")
    
    with gr.Row():
        text_output = gr.Textbox(label="Output Text", interactive=False)
        image_output = gr.Image(label="Generated Image", visible=False)
        audio_output = gr.Audio(label="Generated Audio", visible=False)
    
    # Dynamic UI Updates
    modality.change(
        lambda x: (
            gr.Image(visible=x in ["Image", "Text"]),
            gr.Audio(visible=x == "Text-to-Audio")
        ),
        inputs=modality,
        outputs=[image_output, audio_output]
    )
    
    @demo.load
    def init_models():
        logger.info("Initializing core models...")
        model_manager.get_mistral()
        if DEVICE == "cuda":
            model_manager.get_sd()

    submit_btn.click(
        fn=lambda *args: asyncio.run(process_input(*args)),
        inputs=[modality, text_input, image_input, audio_input, pdf_input, url_input, tts_input],
        outputs=[text_output, image_output, audio_output]
    )
# Add missing process_input function



# --- Execution ---
if __name__ == "__main__":
    try:
        demo.queue().launch(
            server_port=7860,
            show_error=True,
            share=False
        )
    finally:
        model_manager.cleanup()
        logger.info("Cleanup completed")