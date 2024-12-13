import speech_recognition as sr

def extract_audio_data(file_path):
    """
    Extract text from audio files using speech recognition.

    Args:
        file_path (str): Path to the audio file.

    Returns:
        dict: A dictionary with keys `text`, `tables`, and `figures`.
    """
    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(file_path) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
        return {"text": [text], "tables": [], "figures": []}

    except Exception as e:
        raise ValueError(f"Error extracting data from audio file: {e}")
