from reasonchain.utils.lazy_imports import  pytesseract
from PIL import Image

def extract_image_data(file_path):
    """
    Extract text from images using OCR.

    Args:
        file_path (str): Path to the image file.

    Returns:
        dict: A dictionary with keys `text`, `tables`, and `figures`.
    """
    try:
        text = pytesseract.image_to_string(Image.open(file_path))
        return {"text": [text], "tables": [], "figures": [file_path]}

    except Exception as e:
        raise ValueError(f"Error extracting data from image file: {e}")
