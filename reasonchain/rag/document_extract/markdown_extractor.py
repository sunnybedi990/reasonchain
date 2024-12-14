import os
import re
from reasonchain.utils.lazy_imports import pandas as pd
from io import StringIO  # Correct import for string-based file objects


def extract_markdown_data(file_path, download_path="./markdown_images"):
    """
    Extract text, tables, and images from Markdown files.

    Args:
        file_path (str): Path to the Markdown file.
        download_path (str): Directory to save extracted images.

    Returns:
        dict: A dictionary with keys `text`, `tables`, and `figures`.
    """
    try:
        os.makedirs(download_path, exist_ok=True)

        # Read Markdown content
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.readlines()

        # Extract plain text and headings
        text = []
        for line in content:
            if line.strip().startswith("#"):
                text.append(f"Heading: {line.strip()}")
            elif not line.strip().startswith(("|", "![", "*", "-", "[", ">")):
                text.append(line.strip())

        # Extract tables
        tables = []
        table_lines = []
        is_table = False

        for line in content:
            if line.strip().startswith("|") or re.match(r"^[-\s]+$", line.strip()):
                is_table = True
                table_lines.append(line.strip())
            elif is_table:
                # Parse the table once the block ends
                if table_lines:
                    try:
                        table_text = "\n".join(table_lines)
                        df = pd.read_csv(StringIO(table_text), sep="|", skipinitialspace=True)
                        df = df.dropna(axis=1, how="all")  # Drop empty columns
                        tables.append(df.to_dict(orient="records"))
                    except Exception as e:
                        print(f"Error parsing table: {e}")
                    table_lines = []
                    is_table = False

        # Extract images
        figures = []
        image_pattern = r"!\[.*?\]\((.*?)\)"
        for line in content:
            match = re.findall(image_pattern, line)
            for img_path in match:
                # Save image path if remote or copy locally if local
                img_name = os.path.basename(img_path)
                save_path = os.path.join(download_path, img_name)

                if img_path.startswith("http://") or img_path.startswith("https://"):
                    figures.append(img_path)  # Store the remote URL
                else:
                    try:
                        with open(img_path, "rb") as img_file:
                            with open(save_path, "wb") as save_file:
                                save_file.write(img_file.read())
                        figures.append(save_path)
                    except FileNotFoundError:
                        print(f"Image not found: {img_path}")

        return {"text": text, "tables": tables, "figures": figures}

    except Exception as e:
        raise ValueError(f"Error extracting data from Markdown file: {e}")
