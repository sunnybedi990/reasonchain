from reasonchain.utils.lazy_imports import pandas as pd

def extract_excel_data(file_path):
    """Extract data from an Excel file."""
    try:
        dataframes = pd.read_excel(file_path, sheet_name=None)  # Read all sheets into a dictionary
        extracted_data = {}
        for sheet_name, df in dataframes.items():
            extracted_data[sheet_name] = df.to_dict(orient="records")  # Convert to a list of dictionaries
        return extracted_data
    except Exception as e:
        raise ValueError(f"Error extracting data from Excel file: {e}")

def extract_csv_data(file_path):
    """Extract tabular data from CSV files."""
    try:
        df = pd.read_csv(file_path)
        return df.to_dict(orient="records")  # Convert to list of dictionaries
    except Exception as e:
        raise ValueError(f"Error extracting data from CSV file: {e}")