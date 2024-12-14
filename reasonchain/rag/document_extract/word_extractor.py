from reasonchain.utils.lazy_imports import docx
import os


def extract_word_data(file_path, download_path="./word_images"):
    """
    Extract text, tables, and images from Word documents.

    Args:
        file_path (str): Path to the Word file.
        download_path (str): Path to save extracted images.

    Returns:
        dict: A dictionary with keys `text`, `tables`, and `figures`.
    """
    try:
        document = docx.Document(file_path)
        os.makedirs(download_path, exist_ok=True)

        # Extract text
        text = [p.text for p in document.paragraphs if p.text.strip()]

        # Extract tables
        tables = []
        for table in document.tables:
            table_data = []
            for row in table.rows:
                row_data = [cell.text.strip() for cell in row.cells]
                table_data.append(row_data)
            tables.append(table_data)

        # Extract images
        figures = []
        for rel in document.part.rels.values():
            if "image" in rel.target_ref:
                image_path = rel.target_ref
                image_name = os.path.basename(image_path)
                save_path = os.path.join(download_path, image_name)
                with open(save_path, "wb") as f:
                    f.write(rel.target_part.blob)
                figures.append(save_path)

        return {
            "text": text,
            "tables": tables,
            "figures": figures,
        }

    except Exception as e:
        raise ValueError(f"Error extracting data from Word file: {e}")
