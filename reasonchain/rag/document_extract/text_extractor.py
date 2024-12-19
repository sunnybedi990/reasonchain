from reasonchain.utils.lazy_imports import striprtf

def extract_latex_data(file_path):
    """
    Extract text from LaTeX files.

    Args:
        file_path (str): Path to the LaTeX file.

    Returns:
        dict: A dictionary with keys `text`, `tables`, and `figures`.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        # Remove LaTeX commands
        text = re.sub(r"\\[a-zA-Z]+\{.*?\}", "", content)
        return {"text": [text], "tables": [], "figures": []}

    except Exception as e:
        raise ValueError(f"Error extracting data from LaTeX file: {e}")

def extract_rtf_data(file_path):
    """
    Extract plain text from RTF files.

    Args:
        file_path (str): Path to the RTF file.

    Returns:
        dict: A dictionary with keys `text`, `tables`, and `figures`.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            rtf_content = file.read()
        text = striprtf.striprtf.rtf_to_text(rtf_content)
        return {"text": [text], "tables": [], "figures": []}

    except Exception as e:
        raise ValueError(f"Error extracting data from RTF file: {e}")


def extract_text_data(file_path):
    """Extract data from a plain text file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return text
    except Exception as e:
        raise ValueError(f"Error extracting data from text file: {e}")
