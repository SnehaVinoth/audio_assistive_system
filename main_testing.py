# --- main.py ---
# (This assumes all the other .py files are in the same folder)

from image_caption_generator import get_image_caption
from ocr import get_ocr_text
from audio_input import get_voice_input
from audio_output import speak_response
import ollama  # <-- NEW: Using Ollama

# --- NO API KEY NEEDED FOR OLLAMA ---

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
    print(f"System Prompt: {system_prompt}")
    print(f"User Prompt: {user_prompt}")
    print("-------------------------")

    # --- REAL OLLAMA CALL ---
    try:
        # Send to Ollama. Make sure Ollama is running!
        response = ollama.chat(
            model='llama3',  # Make sure you have this model (e.g., 'ollama pull llama3')
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ]
        )
        final_text = response['message']['content']
        
    except Exception as e:
        print(f"Error communicating with Ollama: {e}")
        final_text = (
            "Sorry, I'm having trouble connecting to my local AI model. "
            "Please make sure Ollama is running and you have pulled a model like 'llama3'."
        )
    # --- END REAL CALL ---
    
    return final_text


# --- THE MAIN EXECUTION LOOP ---
def main():
    # 1. Define the image input
    image_path = "path/to/your/image_with_text.jpg" # Point this to a real image

    # 2. Get User's Voice Query
    speak_response("What would you like to know about the image? Or just stay silent for a description.")
    user_query = get_voice_input() # From audio_input.py

    speak_response("Processing the image... please wait.")

    # 3. Get Initial Caption
    initial_caption = get_image_caption(image_path) # From image_caption_generator.py

    # 4. Get Extracted Text
    extracted_text = get_ocr_text(image_path) # From ocr.py

    # 5. Get LLM Interpretation (The "Hub")
    final_response = get_llm_interpretation(initial_caption, extracted_text, user_query)

    # 6. Deliver Audio Output
    speak_response(final_response) # From audio_output.py


if __name__ == "__main__":
    main()