from ebooklib import epub
from bs4 import BeautifulSoup

def extract_ebook_data(file_path):
    """
    Extract text from EPUB files.

    Args:
        file_path (str): Path to the EPUB file.

    Returns:
        dict: A dictionary with keys `text`, `tables`, and `figures`.
    """
    try:
        book = epub.read_epub(file_path)
        text = []

        for item in book.get_items():
            if item.get_type() == epub.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), "html.parser")
                text.append(soup.get_text(separator="\n"))

        return {"text": text, "tables": [], "figures": []}

    except Exception as e:
        raise ValueError(f"Error extracting data from eBook file: {e}")
