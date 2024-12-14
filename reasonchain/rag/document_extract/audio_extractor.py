from reasonchain.utils.lazy_imports import whisper

def extract_audio_data(file_path):
    """
    Extract text from audio files using Whisper ASR.
    
    Args:
        file_path (str): Path to the audio file.
    
    Returns:
        dict: A dictionary with keys `text`, `tables`, and `figures`.
    """
    try:
        model = whisper.load_model("base")  # Load Whisper model
        
        # Transcribe the entire audio file
        result = model.transcribe(file_path)
        
        return {"text": [result["text"]], "tables": [], "figures": []}
    except Exception as e:
        raise ValueError(f"Error extracting data from audio file using Whisper: {e}")

# Example usage
if __name__ == "__main__":
    audio_path = "reasonchain/rag/document_extract/sample.mp3"
    extracted_data = extract_audio_data(audio_path)
    print("Extracted Text:", extracted_data["text"])
