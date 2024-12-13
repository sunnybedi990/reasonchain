import os
from reasonchain.rag.document_extract.pdf_extractor import preprocess_pdf_content
from reasonchain.rag.document_extract.excel_extractor import extract_excel_data , extract_csv_data
from reasonchain.rag.document_extract.json_extractor import extract_json_data
from reasonchain.rag.document_extract.text_extractor import extract_text_data, extract_rtf_data, extract_latex_data
from reasonchain.rag.document_extract.word_extractor import extract_word_data
from reasonchain.rag.document_extract.html_extractor import extract_html_data
from reasonchain.rag.document_extract.markdown_extractor import extract_markdown_data
from reasonchain.rag.document_extract.video_extractor import extract_video_data
from reasonchain.rag.document_extract.image_extractor import extract_image_data
from reasonchain.rag.document_extract.audio_extractor import extract_audio_data
from reasonchain.rag.document_extract.ebook_extractor import extract_ebook_data
from reasonchain.rag.document_extract.powerpoint_extractor import extract_presentation_data
from dotenv import load_dotenv

# Load environment variables for LlamaParse API access
load_dotenv()
# LlamaParse setup
try:
    from llama_parse import LlamaParse
    from llama_index.core.schema import ImageDocument, TextNode
    from llama_index.core import SimpleDirectoryReader
    llama_available = True
except ImportError:
    llama_available = False

def extract_pdf_with_llama(pdf_path: str, download_path: str):
    """Extract tables and text using LlamaParse."""
    parser = LlamaParse(result_type="text")  # Try using plain_text for broader content capture
    json_objs = parser.get_json_result(pdf_path)
    json_list = json_objs[0]["pages"]

    # Extract text nodes
    text_nodes = [
        TextNode(text=page["text"], metadata={"page": page["page"]})
        for page in json_list
    ]

    # Extract image documents
    image_dicts = parser.get_images(json_objs, download_path=download_path)
    image_documents = [
        ImageDocument(image_path=image_dict["path"]) for image_dict in image_dicts
    ]

    return {
        "text": [node.text for node in text_nodes],
        "tables": [],  # LlamaParse doesn't directly extract tables
        "figures": image_documents,
    }

def extract_data(file_path, use_llama=False, download_path="./images"):
    """
    Route the file to the appropriate extraction function based on its extension.

    Args:
        file_path (str): The path to the file to extract.

    Returns:
        dict: A dictionary with keys `text`, `tables`, and `figures`.

    Raises:
        ValueError: If the file extension is unsupported.
    """
    _, ext = os.path.splitext(file_path.lower())

    print(f"Processing file: {file_path}")
    print(f"Detected file extension: {ext}")

    try:
        if ext == ".pdf":
            
        # Extract content from PDF
            if use_llama and llama_available:
                print("Using LlamaParse for document extraction...")
                return extract_pdf_with_llama(file_path, download_path)
            else:
                print("Using regular extraction methods...")
                print("Extracting content...")
                parsed_texts, figures = preprocess_pdf_content(file_path)
                
            if not parsed_texts:
                print("No content extracted from the PDF.")
                return

            print(f"Extracted {len(parsed_texts)} items from the PDF.")

            return {"text": parsed_texts, "tables": [], "figures": figures}
        
        elif ext in [".xls", ".xlsx"]:
            print("Using Excel extractor...")
            tables = extract_excel_data(file_path)
            return {"text": [], "tables": tables, "figures": []}

        elif ext == ".csv":
            print("Using CSV extractor...")
            tables = extract_csv_data(file_path)
            return {"text": [], "tables": tables, "figures": []}

        elif ext in [".doc", ".docx"]:
            print("Using Word extractor...")
            return extract_word_data(file_path, download_path)

        elif ext in [".html", ".htm"]:
            print("Using HTML extractor...")
            return extract_html_data(file_path)

        elif ext == ".json":
            print("Using JSON extractor...")
            text = extract_json_data(file_path)
            return {"text": text, "tables": [], "figures": []}

        elif ext == ".txt":
            print("Using Text extractor...")
            text = extract_text_data(file_path)
            return {"text": [text], "tables": [], "figures": []}

        elif ext == ".md":
            print("Using Markdown extractor...")
            return extract_markdown_data(file_path)
           
        elif ext in [".ppt", ".pptx"]:
            return extract_presentation_data(file_path, download_path)
        elif ext == ".rtf":
            return extract_rtf_data(file_path)
        elif ext in [".epub", ".mobi"]:
            return extract_ebook_data(file_path)
        elif ext in [".png", ".jpg", ".jpeg", ".tiff"]:
            return extract_image_data(file_path)
        elif ext in [".mp4", ".avi", ".mov"]:
            return extract_video_data(file_path)
        elif ext in [".mp3", ".wav"]:
            return extract_audio_data(file_path)
        elif ext == ".tex":
            return extract_latex_data(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    except Exception as e:
        print(f"Error extracting data from file {file_path}: {e}")
        raise


# Example usage for testing
if __name__ == "__main__":
    file_path = "pdfs/tsla-20240930-gen.pdf"  # Replace with your file path
    try:
        extracted_content = extract_data(file_path, use_llama=False)
        print("Extraction successful!")
        print(extracted_content)
    except ValueError as ve:
        print(f"Unsupported file: {ve}")
    except Exception as ex:
        print(f"An error occurred: {ex}")
