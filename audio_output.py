import pyttsx3

# Initialize the TTS engine
engine = pyttsx3.init()
voices = engine.getProperty('voices')
# You can set a specific voice (e.g., a female voice)
# engine.setProperty('voice', voices[1].id) 
engine.setProperty('rate', 180) # Adjust speech rate

def speak_response(text_to_speak):
    """
    Converts a text string to speech and plays it.
    """
    print(f"Assistant: {text_to_speak}")
    engine.say(text_to_speak)
    engine.runAndWait()

# --- Example Usage ---
if __name__ == "__main__":
    speak_response("Hello, I am ready to assist you.")