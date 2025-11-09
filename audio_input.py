import speech_recognition as sr

def get_voice_input():
    """
    Captures audio from the microphone and converts it to text.
    
    Returns:
        str: The transcribed text, or an empty string if it fails.
    """
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening... (Speak your query)")
        r.pause_threshold = 1
        r.adjust_for_ambient_noise(source) # Important for clarity
        audio = r.listen(source)

    try:
        print("Recognizing...")
        # Use Google's free web speech API
        query = r.recognize_google(audio, language='en-us')
        print(f"User query: {query}\n")
        return query
    except sr.UnknownValueError:
        print("Sorry, I did not understand that.")
        return ""
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return ""