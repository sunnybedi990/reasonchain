from reasonchain.utils.lazy_imports import fitz, camelot, pdfplumber, pdf2image, pytesseract
from PIL import Image
import io

from reasonchain.rag.document_extract.helper import (
    clean_text,
    hybrid_chunking,
    save_chunks_to_file,
    initialize_clip_model,
    process_image_with_clip,
    chunk_text_by_semantics
)


def extract_text_with_fitz(pdf_path):
    """Extract text with fallback to OCR."""
    try:
        document = fitz.open(pdf_path)
        text = ""
        for page in document:
            text += page.get_text("text")
        document.close()

        return clean_text(text)

    except Exception as e:
        print(f"Error with PyMuPDF: {e}.")
        return ""


def ocr_pdf(pdf_path):
    """Perform OCR on non-text PDFs."""
    images = pdf2image.convert_from_path(pdf_path)
    ocr_texts = [pytesseract.image_to_string(image, lang="eng") for image in images]
    return "\n".join(ocr_texts)

def remove_headers_footers(text, threshold=3):
    """Remove repeating headers and footers based on frequency."""
    lines = text.split("\n")
    freq = {}
    for line in lines:
        freq[line] = freq.get(line, 0) + 1
    cleaned_lines = [line for line in lines if freq[line] <= threshold]
    return "\n".join(cleaned_lines)

def remove_headers_footers_advanced(text, page_count):
    """Remove headers and footers by analyzing text repetition across pages."""
    lines = text.split("\n")
    freq = {}
    page_length = len(lines) // page_count

    for i, line in enumerate(lines):
        position = i % page_length  # Position within a page
        if position < 3 or position > page_length - 3:  # Header/Footer range
            freq[line] = freq.get(line, 0) + 1

    cleaned_lines = [line for line in lines if freq.get(line, 0) < page_count // 2]
    return "\n".join(cleaned_lines)

def preprocess_text(text):
    """Preprocess text to clean layout-based artifacts."""
    text = remove_headers_footers(text)
    return clean_text(text)

def extract_figures(pdf_path):
    """Extract images/figures from PDF."""
    document = fitz.open(pdf_path)
    figures = []
    for page in document:
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = document.extract_image(xref)
            image_bytes = base_image["image"]
            figures.append(image_bytes)
    document.close()
    return figures


def process_figures_with_captions(pdf_path):
    """Process images from the PDF and add captions using OCR and CLIP embeddings."""
    clip_model, clip_processor = initialize_clip_model()
    figures = extract_figures(pdf_path)
    processed_figures = []

    for idx, figure_bytes in enumerate(figures):
        caption, clip_embedding = process_image_with_clip(
            clip_model, clip_processor, figure_bytes, idx
        )
        image = Image.open(io.BytesIO(figure_bytes))
        ocr_text = pytesseract.image_to_string(image, lang="eng")
        processed_figures.append(
            {"caption": caption, "embedding": clip_embedding, "ocr_text": ocr_text}
        )

    return processed_figures
def process_figure_with_clip(figure_bytes, figure_index):
    """Generate captions and embeddings for a figure using CLIP."""
    clip_model, clip_processor = initialize_clip_model()
    return process_image_with_clip(clip_model, clip_processor, figure_bytes, figure_index)


def extract_tables(pdf_path):
    """Extract tables using Camelot and fallback to PDFPlumber if needed."""
    table_texts = []

    try:
        tables = camelot.read_pdf(pdf_path, pages="all", flavor="lattice")
        if not tables:
            tables = camelot.read_pdf(pdf_path, pages="all", flavor="stream")

        for table in tables:
            df = table.df
            df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x).fillna("")
            for _, row in df.iterrows():
                row_text = ", ".join([f"{col}: {val}" for col, val in row.items() if val])
                table_texts.append(row_text)

    except Exception as e:
        print(f"Camelot failed: {e}. Falling back to PDFPlumber.")
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    for table in page.extract_tables():
                        for row in table:
                            table_texts.append(", ".join(row))
        except Exception as plumber_e:
            print(f"PDFPlumber also failed: {plumber_e}")

    return table_texts

def extract_and_label_tables(pdf_path):
    """Extract tables and label them based on surrounding text context."""
    document = fitz.open(pdf_path)
    tables_with_labels = []

    for page_num, page in enumerate(document):
        # Extract text for context
        context_text = page.get_text("text")
        context_chunks = chunk_text_by_semantics(context_text)

        # Extract tables using PDFPlumber as fallback
        try:
            tables = camelot.read_pdf(pdf_path, pages=str(page_num + 1), flavor="stream")
        except Exception as e:
            print(f"Camelot failed on page {page_num}: {e}")
            tables = []

        for table in tables:
            # Map table to nearby text chunk
            nearby_context = context_chunks[min(len(context_chunks) - 1, page_num)]
            tables_with_labels.append({"table": table.df.to_string(), "context": nearby_context})

    return tables_with_labels


def extract_and_process_pdf_with_saving(pdf_path):
    """Complete extraction workflow with text chunk saving for comparison."""
    full_text = extract_text_with_fitz(pdf_path)

    # Chunk text using hybrid method
    text_chunks = chunk_text_by_semantics(full_text)
    save_chunks_to_file(text_chunks, "hybrid_chunking")

    # Process figures for captions and embeddings
    figures = process_figures_with_captions(pdf_path)

    return {
        "text_chunks": text_chunks,
        "figures": figures,
    }
def preprocess_pdf_content(pdf_path):
    """Process PDF content by extracting text, tables, and figures."""
    full_text = extract_text_with_fitz(pdf_path)
    tables = extract_tables(pdf_path)
    figures = extract_figures(pdf_path)
    parsed_texts =  chunk_text_by_semantics(preprocess_text(full_text))
    parsed_texts.extend(tables)  # Add tables to parsed content
    return parsed_texts, figures

if __name__ == "__main__":
    pdf_path = "pdfs/tsla-20240930-gen.pdf"
    result = extract_and_process_pdf_with_saving(pdf_path)

    print("Text Chunks:")
    for i, chunk in enumerate(result["text_chunks"]):
        print(f"Chunk {i + 1}: {chunk}")

    print("\nFigures:")
    for i, figure in enumerate(result["figures"]):
        print(f"Figure {i + 1}:")
        print(f"Caption: {figure['caption']}")
        print(f"OCR Text: {figure['ocr_text']}")