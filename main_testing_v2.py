# --- main_fileupload.py ---

from ocr import get_ocr_text
from audio_input import get_voice_input
from audio_output import speak_response
import ollama
import os


# We are importing the ViT-GPT2 baseline, not BLIP
from vit_image_caption_generator import get_caption_baseline


def get_llm_interpretation(caption, ocr_text, query):
    """
    Sends the fused data to a local Ollama LLM and gets a response.
    """
    
    # Define the system prompt (the AI's role)
    system_prompt = (
        "You are an assistive AI for a visually impaired user. "
        "You must answer the user's question or describe a scene based on the context provided. "
        "Be direct, clear, and helpful."
    )
    
    # Build the user's prompt with the provided context
    user_prompt = f"""
Here is the information I have from an image:
- Visual Scene: "{caption}"
- Text in Image: "{ocr_text}"
"""
    
    # Add the specific query if one exists
    if not query:
        # Case 1: No user query
        user_prompt += "\nPlease combine this into a single, helpful description."
    else:
        # Case 2: User asked a question
        user_prompt += f"""
My question is: "{query}"
Please answer my question directly.
"""

    print("--- Sending to Ollama ---")
    print(f"User Prompt: {user_prompt}")
    print("-------------------------")

    try:
        response = ollama.chat(
            model='llama3',
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ]
        )
        final_text = response['message']['content']
        
    except Exception as e:
        print(f"Error communicating with Ollama: {e}")
        final_text = "Sorry, I'm having trouble connecting to my local AI model. Please make sure Ollama is running."
    
    return final_text


# --- THE MAIN EXECUTION LOOP ---
def main():
    # 1. Define the image input
    image_path = "test_main.jpeg" # <-- Point this to a real image
    
    if not os.path.exists(image_path):
        speak_response(f"Error: Image path not found at {image_path}")
        return

    # 2. Get User's Voice Query
    speak_response("What would you like to know about the image? Or just stay silent for a description.")
    user_query = get_voice_input() 

    speak_response("Processing the image... please wait.")

    # 3. Get Initial Caption --- MODIFIED FUNCTION CALL ---
    initial_caption = get_caption_baseline(image_path)
    # --------------------------------------------------

    # 4. Get Extracted Text
    extracted_text = get_ocr_text(image_path) 

    # 5. Get LLM Interpretation (The "Hub")
    final_response = get_llm_interpretation(initial_caption, extracted_text, user_query)

    # 6. Deliver Audio Output
    speak_response(final_response) 


if __name__ == "__main__":
    main()