from reasonchain.utils.lazy_imports import pptx
import os

def extract_presentation_data(file_path, download_path="./presentation_images"):
    """
    Extract text, images, and slide content from presentation files.

    Args:
        file_path (str): Path to the presentation file.
        download_path (str): Directory to save extracted images.

    Returns:
        dict: A dictionary with keys `text`, `tables`, and `figures`.
    """
    try:
        os.makedirs(download_path, exist_ok=True)
        prs = pptx.Presentation(file_path)
        text = []
        figures = []

        # Extract text and images from slides
        for slide in prs.slides:
            for shape in slide.shapes:
                if shape.has_text_frame:
                    text.append(shape.text_frame.text.strip())
                if shape.shape_type == 13:  # Image type
                    image = shape.image
                    image_name = f"slide_{slide.slide_id}_{image.filename}"
                    save_path = os.path.join(download_path, image_name)
                    with open(save_path, "wb") as f:
                        f.write(image.blob)
                    figures.append(save_path)

        return {"text": text, "tables": [], "figures": figures}

    except Exception as e:
        raise ValueError(f"Error extracting data from presentation file: {e}")
