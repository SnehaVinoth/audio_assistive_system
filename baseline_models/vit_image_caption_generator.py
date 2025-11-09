from transformers import VisionEncoderDecoderModel, ViTImageProcessor, GPT2TokenizerFast
from PIL import Image
import torch

# Set up device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the ViT-GPT2 model and its specific processors
model_name = "nlpconnect/vit-gpt2-image-captioning"
model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
image_processor = ViTImageProcessor.from_pretrained(model_name)
tokenizer = GPT2TokenizerFast.from_pretrained(model_name)

def get_caption_baseline(image_path):
    """
    Generates a caption for a given image using the ViT-GPT2 baseline.
    This represents the "classic" simple encoder-decoder architecture.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        
        # Process the image
        pixel_values = image_processor(images=image, return_tensors="pt").pixel_values.to(device)
        
        # Generate token IDs
        # Using num_beams=4 (beam search) often gives better results
        output_ids = model.generate(pixel_values, max_new_tokens=50, num_beams=4)
        
        # Decode the IDs into a text caption
        caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return caption

    except Exception as e:
        print(f"Error generating baseline caption: {e}")
        return "Could not generate baseline caption."

# --- Example Usage ---
if __name__ == "__main__":
    # Test this with an image from your Flickr8k dataset
    test_image_path = "baseline_models/test_image.jpeg" 
    
    baseline_caption = get_caption_baseline(test_image_path)
    print("--- Baseline (ViT-GPT2) Caption ---")
    print(baseline_caption)