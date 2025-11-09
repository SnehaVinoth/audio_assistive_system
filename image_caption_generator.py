from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Set up the device (use GPU if available, otherwise CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the pre-trained model and processor
# Using the "large" model for better accuracy
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

def get_image_caption(image_path):
    """
    Generates a caption for a given image.
    
    Args:
        image_path (str): The file path to the image.
        
    Returns:
        str: The generated caption.
    """
    try:
        # Load and preprocess the image
        raw_image = Image.open(image_path).convert('RGB')
        
        # You can add "a photography of" to guide the model, but it's optional
        # text = "a photography of"
        # inputs = processor(raw_image, text, return_tensors="pt").to(device)
        
        # Process image without text prompt for a general caption
        inputs = processor(raw_image, return_tensors="pt").to(device)

        # Generate the caption
        out = model.generate(**inputs, max_new_tokens=50) # Increased token limit
        
        # Decode the caption
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    
    except Exception as e:
        print(f"Error generating caption: {e}")
        return "Could not generate a caption for the image."

# --- Example Usage ---
if __name__ == "__main__":
    # Test this with any image file, e.g., one from your Flickr8k folder
    # 
    #test_image_path = "path/to/your/image.jpg" 
    
   # initial_caption = get_image_caption(test_image_path)
    print("--- Initial Caption ---")
    #print(initial_caption)