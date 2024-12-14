from reasonchain.utils.lazy_imports import bs4, pandas as pd, requests, os

def extract_html_data(file_path, download_path="./html_images"):
    """
    Extract text, tables, and images from HTML files.

    Args:
        file_path (str): Path to the HTML file.
        download_path (str): Directory to save extracted images.

    Returns:
        dict: A dictionary with keys `text`, `tables`, and `figures`.
    """
    try:
        os.makedirs(download_path, exist_ok=True)

        # Read and parse the HTML file
        with open(file_path, "r", encoding="utf-8") as file:
            soup = bs4.BeautifulSoup(file, "html.parser")

        # Extract text
        text = soup.get_text(separator="\n").strip()

        # Extract tables
        tables = []
        for table in soup.find_all("table"):
            try:
                df = pd.read_html(str(table))[0]  # Parse table with pandas
                tables.append(df.to_dict(orient="records"))  # Convert to list of dicts
            except ValueError:
                print("Skipping a table that couldn't be parsed.")

        # Extract images
        figures = []
        for img_tag in soup.find_all("img"):
            img_url = img_tag.get("src")
            if img_url:
                # Save images locally
                img_name = os.path.basename(img_url)
                save_path = os.path.join(download_path, img_name)

                # Check if the image source is a URL or a local path
                if img_url.startswith("http://") or img_url.startswith("https://"):
                    response = requests.get(img_url)
                    with open(save_path, "wb") as img_file:
                        img_file.write(response.content)
                else:
                    with open(img_url, "rb") as img_file:
                        with open(save_path, "wb") as save_file:
                            save_file.write(img_file.read())

                figures.append(save_path)

        return {"text": [text], "tables": tables, "figures": figures}

    except Exception as e:
        raise ValueError(f"Error extracting data from HTML file: {e}")
