from reasonchain.utils.lazy_imports import moviepy
from PIL import Image
def extract_video_data(file_path):
    """
    Extract frames and text from video files.

    Args:
        file_path (str): Path to the video file.

    Returns:
        dict: A dictionary with keys `text`, `tables`, and `figures`.
    """
    try:
        clip = moviepy.video.io.VideoFileClip(file_path)
        figures = []

        for i, frame in enumerate(clip.iter_frames(fps=1)):  # 1 frame per second
            image_path = f"{file_path}_frame_{i}.png"
            Image.fromarray(frame).save(image_path)
            figures.append(image_path)

        return {"text": [], "tables": [], "figures": figures}

    except Exception as e:
        raise ValueError(f"Error extracting data from video file: {e}")
