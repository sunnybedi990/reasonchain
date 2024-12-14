import re
from reasonchain.utils.lazy_imports import transformers, spacy, os
from PIL import Image
import io


def load_spacy_model(model_name="en_core_web_sm"):
    """Load a Spacy model, downloading it if not already available."""
    try:
        return spacy.load(model_name)
    except OSError:
        print(f"Spacy model '{model_name}' not found. Downloading...")
        spacy.cli.download(model_name)
        return spacy.load(model_name)


# Initialize NLP
nlp = load_spacy_model()


def clean_text(text):
    """Clean text by removing extraneous symbols and whitespace."""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\u2022", "-", text)  # Replace bullet points
    return text


def chunk_text(text, max_length=512):
    """Chunk text into manageable pieces for embeddings."""
    doc = nlp(text)
    chunks, chunk = [], []
    length = 0

    for sent in doc.sents:
        length += len(sent.text)
        if length > max_length:
            chunks.append(" ".join(chunk))
            chunk = []
            length = len(sent.text)
        chunk.append(sent.text)

    if chunk:
        chunks.append(" ".join(chunk))
    return chunks

def chunk_by_topics(text, max_length=512):
    """Chunk text based on topics and semantic structure."""
    doc = nlp(text)
    chunks, current_chunk = [], []
    current_length = 0

    for sentence in doc.sents:
        if current_length + len(sentence.text) > max_length or "Topic:" in sentence.text:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(sentence.text)
        current_length += len(sentence.text)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def chunk_text_advanced(text, max_length=512, split_on=["\n\n", ". ", "? ", "! "]):
    """Chunk text into meaningful pieces based on topic, paragraph, and sentence boundaries."""
    chunks, current_chunk = [], ""
    current_length = 0

    for delimiter in split_on:
        segments = text.split(delimiter)
        for segment in segments:
            segment = segment.strip()
            if current_length + len(segment) > max_length:
                chunks.append(current_chunk.strip())
                current_chunk = ""
                current_length = 0
            current_chunk += segment + delimiter
            current_length += len(segment) + len(delimiter)

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def chunk_text_by_semantics(text, max_length=512):
    """Chunk text into meaningful semantic units, respecting topic and paragraph boundaries."""
    doc = nlp(text)
    chunks, current_chunk = [], []
    current_length = 0

    for sentence in doc.sents:
        # Check for topic headers or new paragraphs
        if current_length + len(sentence.text) > max_length or any(
            keyword in sentence.text.lower() for keyword in ["introduction", "conclusion", "summary", "table of contents"]
        ):
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(sentence.text)
        current_length += len(sentence.text)

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def hybrid_chunking(text, max_length=512, advanced_split_on=["\n\n", ". ", "? ", "! "]):
    """
    Hybrid chunking function that combines semantic and advanced chunking strategies.
    """
    num_sentences = len(list(nlp(text).sents))
    avg_sentence_length = sum(len(sent.text) for sent in nlp(text).sents) / num_sentences

    sentence_threshold = 100  # Number of sentences to switch to advanced chunking
    avg_length_threshold = 20  # Average sentence length to use semantic chunking

    if num_sentences > sentence_threshold or avg_sentence_length < avg_length_threshold:
        print("Using advanced chunking.")
        return chunk_text_advanced(text, max_length, split_on=advanced_split_on)
    else:
        print("Using semantic chunking.")
        return chunk_text(text, max_length=max_length)


def save_chunks_to_file(chunks, method_name, output_dir="parsed_chunks"):
    """Save text chunks to a file for comparison."""
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
    file_path = os.path.join(output_dir, f"{method_name}_chunks.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks, 1):
            f.write(f"Chunk {i}:\n{chunk}\n\n")
    print(f"Chunks saved to {file_path}")


def initialize_clip_model():
    """Load the CLIP model and processor."""
    clip_model = transformers.CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = transformers.CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return clip_model, clip_processor


def process_image_with_clip(clip_model, clip_processor, image_bytes, figure_index):
    """Generate captions and embeddings for an image using CLIP."""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt")
        embeddings = clip_model.get_image_features(**inputs).detach().numpy()
        caption = f"Figure {figure_index + 1}: Semantic embedding added."
        return caption, embeddings
    except Exception as e:
        print(f"Error processing figure {figure_index + 1} with CLIP: {e}")
        return f"Figure {figure_index + 1}: Unable to process.", None
