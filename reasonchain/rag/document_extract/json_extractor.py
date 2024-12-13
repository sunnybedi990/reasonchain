import json

def extract_json_data(file_path):
    """Extract data from a JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise ValueError(f"Error extracting data from JSON file: {e}")
