import easyocr
import torch

# Initialize the EasyOCR reader
# This will download the model the first time it's run
# We specify 'en' for English
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

def get_ocr_text(image_path):
    """
    Extracts all text from a given image.
    
    Args:
        image_path (str): The file path to the image.
        
    Returns:
        str: A single string containing all detected text, separated by spaces.
    """
    try:
        # Read text from the image
        # 'detail=0' returns only the text, not coordinates
        # 'paragraph=True' joins text blocks that are close together
        result = reader.readtext(image_path, detail=0, paragraph=True)
        
        # Join all detected text fragments into a single string
        if result:
            return " ".join(result)
        else:
            return "No text detected."
            
    except Exception as e:
        print(f"Error extracting text: {e}")
        return "Error during text extraction."


# if __name__ == "__main__":

#     test_image_path = "" 
    
#     extracted_text = get_ocr_text(test_image_path)
#     print("--- Extracted Text ---")
#     print(extracted_text)