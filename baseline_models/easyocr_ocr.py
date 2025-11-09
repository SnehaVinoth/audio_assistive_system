import easyocr
import torch

# Initialize the EasyOCR reader (this will download the model)
# We can tell it to use the GPU if available.
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

def get_ocr_text(image_path):
    """
    Extracts all text from a given image using EasyOCR.
    """
    try:
        # Read text from the image
        # 'detail=0' returns only text
        # 'paragraph=True' joins nearby text blocks
        result = reader.readtext(image_path, detail=0, paragraph=True)
        
        if result:
            return " ".join(result)
        else:
            return "No text detected."
            
    except Exception as e:
        print(f"Error extracting text: {e}")
        return "Error during text extraction."

# --- Example Usage ---
if __name__ == "__main__":
    # Use an image with challenging, "in-the-wild" text
    test_image_path = "baseline_models/test_text.jpeg" # Use the SAME image as the baseline
    
    text = get_ocr_text(test_image_path)
    print("--- EasyOCR (Your Model) Text ---")
    print(text)