# Standard library imports
import io
import numpy as np
from PIL import Image

# Local imports
from reasonchain.rag.document_extract.helper import (
    clean_text,
    hybrid_chunking,
    save_chunks_to_file,
    initialize_clip_model,
    process_image_with_clip,
    chunk_text_by_semantics
)
from reasonchain.rag.vector.utils import resize_embeddings

def extract_text_with_fitz(pdf_path):
    """Extract text with fallback to OCR."""
    try:
        from reasonchain.utils.lazy_imports import fitz
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
    from reasonchain.utils.lazy_imports import pdf2image, pytesseract
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
    from reasonchain.utils.lazy_imports import fitz
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


def process_figures_batch(figures, start_idx=0, target_dim=384):
    """Process a batch of figures using CLIP for better performance.
    
    Args:
        figures (list): List of figure byte arrays to process
        start_idx (int): Starting index for figure numbering
        target_dim (int): Target dimensionality for embeddings
        
    Returns:
        tuple: (list of captions, array of embeddings)
    """
    clip_model, clip_processor = initialize_clip_model()
    captions = []
    all_embeddings = []
    
    try:
        # Process all figures in the batch
        for idx, figure_bytes in enumerate(figures):
            caption, clip_embedding = process_image_with_clip(
                clip_model, clip_processor, figure_bytes, start_idx + idx
            )
            captions.append(caption)
            if clip_embedding is not None:
                # Ensure embedding is 2D array with shape (1, embedding_dim)
                if len(clip_embedding.shape) == 1:
                    clip_embedding = clip_embedding.reshape(1, -1)
                all_embeddings.append(clip_embedding)
        
        # Stack all embeddings if we have any
        if all_embeddings:
            # Stack embeddings first
            stacked_embeddings = np.vstack(all_embeddings)
            # Resize to target dimension if needed
            if stacked_embeddings.shape[1] != target_dim:
                stacked_embeddings = resize_embeddings(stacked_embeddings, target_dim=target_dim)
            return captions, stacked_embeddings
        return captions, None
        
    except Exception as e:
        print(f"Error processing figure batch: {e}")
        return [], None

def process_figures_with_captions(pdf_path, batch_size=32):
    """Process images from the PDF and add captions using OCR and CLIP embeddings."""
    from reasonchain.utils.lazy_imports import fitz, pdf2image, pytesseract
    figures = extract_figures(pdf_path)
    all_processed_figures = []
    
    # Process figures in batches
    for i in range(0, len(figures), batch_size):
        batch_figures = figures[i:i + batch_size]
        captions, embeddings = process_figures_batch(batch_figures, start_idx=i)
        
        # Process OCR in parallel for the batch
        for idx, figure_bytes in enumerate(batch_figures):
            image = Image.open(io.BytesIO(figure_bytes))
            ocr_text = pytesseract.image_to_string(image, lang="eng")
            
            all_processed_figures.append({
                "caption": captions[idx] if idx < len(captions) else "",
                "embedding": embeddings[idx] if embeddings is not None and idx < len(embeddings) else None,
                "ocr_text": ocr_text
            })
    
    return all_processed_figures
def process_figure_with_clip(figure_bytes, figure_index):
    """Generate captions and embeddings for a figure using CLIP."""
    clip_model, clip_processor = initialize_clip_model()
    return process_image_with_clip(clip_model, clip_processor, figure_bytes, figure_index)


def extract_tables_with_camelot(pdf_path):
    """Extract tables using Camelot."""
    from reasonchain.utils.lazy_imports import camelot
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
                if row_text.strip():  # Only add non-empty rows
                    table_texts.append(row_text)
        return table_texts, None
    except Exception as e:
        return None, str(e)

def extract_tables_with_pdfplumber(pdf_path, camelot_error=None):
    """Extract tables using PDFPlumber as fallback."""
    from reasonchain.utils.lazy_imports import pdfplumber
    table_texts = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        if table:  # Check if table exists
                            for row in table:
                                if row:  # Check if row exists
                                    # Filter out None values and convert to strings
                                    cleaned_row = [str(cell).strip() if cell is not None else "" for cell in row]
                                    # Only add non-empty rows
                                    if any(cleaned_row):
                                        table_texts.append(", ".join(cleaned_row))
        return table_texts
    except Exception as e:
        print(f"PDFPlumber failed after Camelot error: {camelot_error}\nPDFPlumber error: {e}")
        return []

def extract_tables(pdf_path):
    """Extract tables using Camelot with PDFPlumber fallback."""
    # Try Camelot first
    table_texts, camelot_error = extract_tables_with_camelot(pdf_path)
    
    # If Camelot fails or returns no tables, try PDFPlumber
    if table_texts is None or not table_texts:
        print(f"Camelot failed or found no tables: {camelot_error}. Falling back to PDFPlumber.")
        table_texts = extract_tables_with_pdfplumber(pdf_path, camelot_error)
    
    return table_texts
    
def extract_and_label_tables(pdf_path):
    """Extract tables and label them based on surrounding text context."""
    from reasonchain.utils.lazy_imports import fitz, camelot
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
    parsed_texts = chunk_text_by_semantics(preprocess_text(full_text))
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