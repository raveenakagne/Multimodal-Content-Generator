import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import matplotlib.pyplot as plt

# Set device to MPS if available, else CPU
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Load BLIP model and processor
model_id = "Salesforce/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(model_id)
model = BlipForConditionalGeneration.from_pretrained(model_id).to(device)

def generate_image_caption(image_path):
    """
    Generates a caption for an image using the BLIP model.
    
    Parameters:
    - image_path (str): The path to the image file.

    Returns:
    - str: Generated caption.
    - PIL.Image: The original image.
    """
    # Load and process the image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    # Generate caption
    with torch.no_grad():
        output = model.generate(**inputs, max_length=50, num_beams=5)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption, image

def display_captioned_image(image_path):
    """
    Displays the image alongside the generated caption.

    Parameters:
    - image_path (str): The path to the image file.
    """
    caption, img = generate_image_caption(image_path)
    
    # Display image and caption
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Original Image')
    
    plt.subplot(1, 2, 2)
    plt.text(0.05, 0.5, caption, wrap=True, fontsize=10)
    plt.axis('off')
    plt.title('Generated Caption')
    
    plt.tight_layout()
    plt.show()
    
    print("Image Description:")
    print(caption)
    print("-" * 100)

# Example usage
image_path = "example_flicker8kimages/98944492.jpg"
display_captioned_image(image_path)
