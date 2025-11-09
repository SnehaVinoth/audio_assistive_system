from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Set up the device (use GPU if available, otherwise CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the PRE-TRAINED BLIP MODEL and its unified processor
# This will download the model (about 1.76 GB) the first time
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

def get_image_caption(image_path):
    """
    Generates a caption for a given image using the BLIP model.
    """
    try:
        # Load and preprocess the image
        raw_image = Image.open(image_path).convert('RGB')
        
        # Process image for a general caption
        # This prepares the image for the model
        inputs = processor(raw_image, return_tensors="pt").to(device)

        # Generate the caption
        out = model.generate(**inputs, max_new_tokens=50) 
        
        # Decode the caption
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    
    except Exception as e:
        print(f"Error generating caption: {e}")
        return "Could not generate a caption for the image."

# --- Example Usage ---
if __name__ == "__main__":

    test_image_path = "baseline_models/test_image.jpeg" 
    
    print(f"Loading BLIP model and processing image at: {test_image_path}...")
    
    # Run the captioning function
    caption = get_image_caption(test_image_path)
    
    print("\n" + "="*30)
    print("--- BLIP (Your Model) Caption ---")
    print(caption)