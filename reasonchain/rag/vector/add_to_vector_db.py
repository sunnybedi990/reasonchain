import os
from reasonchain.rag.vector.VectorDB import VectorDB
from reasonchain.rag.document_extract.pdf_extractor import process_figure_with_clip
from reasonchain.rag.document_extract.extractor_handler import extract_data

def add_pdf_to_vector_db(
    file_path,
    db_path='vector_db.index',
    db_type='faiss',
    db_config='true',
    embedding_provider='sentence_transformers',
    embedding_model='all-mpnet-base-v2',
    use_gpu=True,
    use_llama=False,
    api_key=None
):
    
    """
    Processes a file, extracts content, and adds it to a vector database.

    Args:
        file_path (str): Path to the file to process.
        db_path (str): Path to the vector database index.
        db_type (str): Type of vector database (e.g., "faiss").
        db_config (str): Configuration for the vector database.
        embedding_provider (str): Embedding provider to use.
        embedding_model (str): Embedding model to use.
        use_gpu (bool): Whether to use GPU for embedding.
        use_llama (bool): Whether to use LlamaParse for PDF extraction.
        api_key (str): API key for vector database if required.
    """
    try:
        
        extracted_data = extract_data(file_path, use_llama)
        parsed_texts = extracted_data.get("text", [])
        tables = extracted_data.get("tables", [])
        figures = extracted_data.get("figures", [])

        if not parsed_texts and not tables:
            print("No content extracted from the file.")
            return

        # Combine text and tables for embedding
        all_content = parsed_texts + [str(table) for table in tables]


        print(f"Extracted {len(all_content)} items from the PDF.")

        # Initialize the vector database
        db = VectorDB(
            db_path=db_path,
            db_type=db_type,
            db_config=db_config,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            use_gpu=use_gpu,
            api_key=api_key,
            collection_name=os.path.splitext(os.path.basename(db_path))[0],  # Milvus-specific
            index_name=os.path.splitext(os.path.basename(db_path))[0]  # Pinecone-specific
        )
            # Process figures using CLIP
        if figures:
            print(f"Extracted {len(figures)} figures from the PDF.")
            for i, figure in enumerate(figures):
                caption, clip_embedding = process_figure_with_clip(figure, i)

                # Append caption to parsed_texts
                all_content.append(caption)

                # Add figure embedding directly to the vector database
                if clip_embedding is not None:
                    print(clip_embedding.shape)
                    db.add_embeddings(clip_embeddings=clip_embedding, texts=[caption])


        # Add embeddings to the vector database
        db.add_embeddings(all_content)

        # Save the FAISS index if applicable
        if db_type == "faiss":
            db.save_index(db_path)
            print(f"FAISS index saved at {db_path}.")
        else:
            print(f"Data added to {db_type} vector database.")

    except Exception as e:
        print(f"Error adding PDF to vector database: {e}")
        raise


# Usage Example:
if __name__ == "__main__":
    pdf_path = 'example.pdf'
    add_pdf_to_vector_db(pdf_path, db_path='example_vector_db.index', use_llama=True)
