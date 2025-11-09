import pytesseract
from PIL import Image

def get_ocr_tesseract(image_path):
    """
    Extracts text from an image using the Tesseract baseline.
    """
    try:
        # Open the image
        image = Image.open(image_path)
        
        # Run Tesseract OCR on the imagea
        text = pytesseract.image_to_string(image)
        
        return text.strip() if text.strip() else "No text detected."
            
    except Exception as e:
        print(f"Error with Tesseract: {e}")
        # This can fail if the tesseract command isn't found
        return "Error during Tesseract processing."

# --- Example Usage ---
if __name__ == "__main__":
    # Use an image that has text, preferably a street sign or product
    test_image_path = "baseline_models/test_text.jpeg" 
    
    baseline_text = get_ocr_tesseract(test_image_path)
    print("--- Tesseract Baseline OCR ---")
    print(baseline_text)