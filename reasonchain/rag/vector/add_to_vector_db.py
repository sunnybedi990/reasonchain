import os
# Set tokenizer parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from reasonchain.rag.vector.VectorDB import VectorDB
from reasonchain.rag.document_extract.pdf_extractor import process_figure_with_clip, process_figures_batch
from reasonchain.rag.document_extract.extractor_handler import extract_data
from reasonchain.utils.lazy_imports import psycopg2, elasticsearch, requests

def add_raw_data_to_vector_db(
    texts=None,
    embeddings=None,
    metadata=None,
    db_path='vector_db.index',
    db_type='faiss',
    db_config=None,
    embedding_provider='sentence_transformers',
    embedding_model='all-mpnet-base-v2',
    use_gpu=True,
    api_key=None,
    batch_size=32,
    create_embeddings=True
):
    """
    Add raw text data, pre-computed embeddings, or structured data directly to a vector database.
    
    This function allows you to add data that has already been extracted or processed elsewhere,
    without needing to go through file processing.
    
    Args:
        texts (list, optional): List of text strings to add to the database
        embeddings (np.array or list, optional): Pre-computed embeddings array/list
        metadata (list, optional): List of metadata dictionaries for each text/embedding
        db_path (str): Path to the vector database index
        db_type (str): Type of vector database ('faiss', 'milvus', 'pinecone', 'qdrant', 'weaviate')
        db_config (str): Configuration for the vector database
        embedding_provider (str): Embedding provider to use (if generating embeddings)
        embedding_model (str): Embedding model to use (if generating embeddings)
        use_gpu (bool): Whether to use GPU for embedding generation
        api_key (str): API key for vector database if required
        batch_size (int): Size of batches for processing
        create_embeddings (bool): Whether to generate embeddings from texts (if embeddings not provided)
        
    Returns:
        Result from the vector database addition operation
        
    Raises:
        ValueError: If neither texts nor embeddings are provided
        Exception: If there's an error during database operations
        
    Examples:
        # Add raw text data (embeddings will be generated)
        texts = ["This is document 1", "This is document 2"]
        add_raw_data_to_vector_db(texts=texts)
        
        # Add pre-computed embeddings with texts
        import numpy as np
        embeddings = np.random.rand(2, 768)  # 2 embeddings of 768 dimensions
        texts = ["Text 1", "Text 2"]
        add_raw_data_to_vector_db(texts=texts, embeddings=embeddings)
        
        # Add with custom metadata
        texts = ["Document 1", "Document 2"]
        metadata = [{"source": "api", "type": "news"}, {"source": "web", "type": "blog"}]
        add_raw_data_to_vector_db(texts=texts, metadata=metadata)
    """
    try:
        # Validate inputs
        if texts is None and embeddings is None:
            raise ValueError("Either 'texts' or 'embeddings' must be provided")
        
        if texts is not None:
            if not isinstance(texts, list):
                texts = [texts]  # Convert single text to list
            print(f"Processing {len(texts)} text entries")
        
        if embeddings is not None:
            import numpy as np
            if not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings, dtype='float32')
            print(f"Processing {len(embeddings)} pre-computed embeddings")
        
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
        
        # Handle different input scenarios
        if texts is not None and embeddings is None and create_embeddings:
            # Generate embeddings from texts
            print("Generating embeddings from provided texts...")
            result = db.add_embeddings(texts)
        elif texts is not None and embeddings is not None:
            # Use provided embeddings with texts
            print("Using provided embeddings with texts...")
            result = db.add_embeddings(texts, embeddings=embeddings)
        elif texts is None and embeddings is not None:
            # Use embeddings only (generate placeholder texts)
            placeholder_texts = [f"embedding_{i}" for i in range(len(embeddings))]
            print("Using embeddings with generated placeholder texts...")
            result = db.add_embeddings(placeholder_texts, embeddings=embeddings)
        else:
            raise ValueError("Invalid combination of parameters")
        
        # Save the FAISS index if applicable
        if db_type == "faiss":
            db.save_index(db_path)
            print(f"FAISS index saved at {db_path}")
        else:
            print(f"Data added to {db_type} vector database")
        
        return result
        
    except Exception as e:
        print(f"Error adding raw data to vector database: {e}")
        raise

def add_structured_data_to_vector_db(
    data_dict,
    text_field='text',
    embedding_field=None,
    metadata_fields=None,
    db_path='vector_db.index',
    db_type='faiss',
    db_config=None,
    embedding_provider='sentence_transformers',
    embedding_model='all-mpnet-base-v2',
    use_gpu=True,
    api_key=None,
    batch_size=32
):
    """
    Add structured data (like JSON, DataFrame-like structures) to a vector database.
    
    Args:
        data_dict (list of dict or dict): Structured data where each item/row contains text and metadata
        text_field (str): Field name containing the text content
        embedding_field (str, optional): Field name containing pre-computed embeddings
        metadata_fields (list, optional): List of field names to include as metadata
        db_path (str): Path to the vector database index
        db_type (str): Type of vector database
        db_config (str): Configuration for the vector database
        embedding_provider (str): Embedding provider to use
        embedding_model (str): Embedding model to use
        use_gpu (bool): Whether to use GPU for embedding generation
        api_key (str): API key for vector database if required
        batch_size (int): Size of batches for processing
        
    Returns:
        Result from the vector database addition operation
        
    Examples:
        # Add structured data from API response
        data = [
            {"text": "Article 1 content", "title": "Article 1", "category": "tech", "date": "2025-01-01"},
            {"text": "Article 2 content", "title": "Article 2", "category": "science", "date": "2025-01-02"}
        ]
        add_structured_data_to_vector_db(data, metadata_fields=['title', 'category', 'date'])
        
        # Add data with pre-computed embeddings
        data_with_embeddings = [
            {"text": "Content 1", "embedding": [0.1, 0.2, ...], "source": "web"},
            {"text": "Content 2", "embedding": [0.3, 0.4, ...], "source": "api"}
        ]
        add_structured_data_to_vector_db(
            data_with_embeddings, 
            embedding_field='embedding',
            metadata_fields=['source']
        )
    """
    try:
        # Normalize input to list of dictionaries
        if isinstance(data_dict, dict):
            data_dict = [data_dict]
        
        if not isinstance(data_dict, list):
            raise ValueError("data_dict must be a list of dictionaries or a single dictionary")
        
        print(f"Processing {len(data_dict)} structured data entries")
        
        # Extract texts
        texts = []
        embeddings = None
        metadata_list = []
        
        for i, item in enumerate(data_dict):
            if not isinstance(item, dict):
                raise ValueError(f"Item {i} is not a dictionary")
            
            # Extract text
            if text_field not in item:
                raise ValueError(f"Item {i} missing required field '{text_field}'")
            texts.append(item[text_field])
            
            # Extract metadata if specified
            if metadata_fields:
                metadata = {field: item.get(field, None) for field in metadata_fields}
                metadata_list.append(metadata)
        
        # Extract embeddings if specified
        if embedding_field:
            import numpy as np
            embeddings_list = []
            for i, item in enumerate(data_dict):
                if embedding_field in item:
                    embeddings_list.append(item[embedding_field])
                else:
                    print(f"Warning: Item {i} missing embedding field '{embedding_field}'")
            
            if embeddings_list:
                embeddings = np.array(embeddings_list, dtype='float32')
                print(f"Found {len(embeddings)} pre-computed embeddings")
        
        # Use the raw data function to add to database
        return add_raw_data_to_vector_db(
            texts=texts,
            embeddings=embeddings,
            metadata=metadata_list,
            db_path=db_path,
            db_type=db_type,
            db_config=db_config,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            use_gpu=use_gpu,
            api_key=api_key,
            batch_size=batch_size
        )
        
    except Exception as e:
        print(f"Error adding structured data to vector database: {e}")
        raise

def add_external_source_to_vector_db(
    source_type,
    source_config,
    db_path='vector_db.index',
    db_type='faiss',
    db_config=None,
    embedding_provider='sentence_transformers',
    embedding_model='all-mpnet-base-v2',
    use_gpu=True,
    api_key=None,
    batch_size=32,
    limit=None
):
    """
    Add data from external sources (databases, APIs, cloud storage) to a vector database.
    
    Args:
        source_type (str): Type of source ('database', 'api', 'cloud_storage', 'elasticsearch')
        source_config (dict): Configuration for the external source
        db_path (str): Path to the vector database index
        db_type (str): Type of vector database
        db_config (str): Configuration for the vector database
        embedding_provider (str): Embedding provider to use
        embedding_model (str): Embedding model to use
        use_gpu (bool): Whether to use GPU for embedding generation
        api_key (str): API key for vector database if required
        batch_size (int): Size of batches for processing
        limit (int, optional): Maximum number of records to process
        
    Returns:
        Result from the vector database addition operation
        
    Examples:
        # Add from database
        db_config = {
            'connection_string': 'postgresql://user:pass@localhost/db',
            'query': 'SELECT text, title, category FROM articles WHERE published = true',
            'text_column': 'text',
            'metadata_columns': ['title', 'category']
        }
        add_external_source_to_vector_db('database', db_config)
        
        # Add from REST API
        api_config = {
            'url': 'https://api.example.com/articles',
            'headers': {'Authorization': 'Bearer token'},
            'text_field': 'content',
            'metadata_fields': ['title', 'author', 'date']
        }
        add_external_source_to_vector_db('api', api_config)
    """
    try:
        print(f"Fetching data from {source_type} source...")
        
        data = []
        
        if source_type == 'database':
            data = _fetch_from_database(source_config, limit)
        elif source_type == 'api':
            data = _fetch_from_api(source_config, limit)
        elif source_type == 'cloud_storage':
            data = _fetch_from_cloud_storage(source_config, limit)
        elif source_type == 'elasticsearch':
            data = _fetch_from_elasticsearch(source_config, limit)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
        
        if not data:
            print("No data fetched from external source")
            return None
        
        print(f"Fetched {len(data)} records from {source_type}")
        
        # Use structured data function to add to database
        return add_structured_data_to_vector_db(
            data,
            text_field=source_config.get('text_field', 'text'),
            embedding_field=source_config.get('embedding_field'),
            metadata_fields=source_config.get('metadata_fields'),
            db_path=db_path,
            db_type=db_type,
            db_config=db_config,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            use_gpu=use_gpu,
            api_key=api_key,
            batch_size=batch_size
        )
        
    except Exception as e:
        print(f"Error adding data from external source: {e}")
        raise

def _fetch_from_database(config, limit=None):
    """Fetch data from a database using SQL query."""
    try:
        import sqlite3   # SQLite (built-in module)
        from urllib.parse import urlparse
        
        connection_string = config['connection_string']
        query = config['query']
        
        if limit:
            query += f" LIMIT {limit}"
        
        # Parse connection string to determine database type
        parsed = urlparse(connection_string)
        
        data = []
        if parsed.scheme in ['postgresql', 'postgres']:
            conn = psycopg2.connect(connection_string)
            cursor = conn.cursor()
            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            data = [dict(zip(columns, row)) for row in rows]
            conn.close()
        elif parsed.scheme == 'sqlite':
            conn = sqlite3.connect(parsed.path)
            cursor = conn.cursor()
            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            data = [dict(zip(columns, row)) for row in rows]
            conn.close()
        else:
            raise ValueError(f"Unsupported database type: {parsed.scheme}")
        
        return data
        
    except ImportError as e:
        print(f"Database connector not available: {e}")
        print("Install required packages: pip install psycopg2-binary (PostgreSQL) or use sqlite3 (built-in)")
        return []
    except Exception as e:
        print(f"Database fetch error: {e}")
        return []

def _fetch_from_api(config, limit=None):
    """Fetch data from a REST API."""
    try:
        url = config['url']
        headers = config.get('headers', {})
        params = config.get('params', {})
        
        if limit:
            params['limit'] = limit
        
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        # Handle different API response formats
        if isinstance(data, dict):
            if 'data' in data:
                return data['data']
            elif 'results' in data:
                return data['results']
            elif 'items' in data:
                return data['items']
            else:
                return [data]  # Single item
        elif isinstance(data, list):
            return data
        else:
            return [data]
            
    except ImportError:
        print("requests library not available. Install with: pip install requests")
        return []
    except Exception as e:
        print(f"API fetch error: {e}")
        return []

def _fetch_from_cloud_storage(config, limit=None):
    """Fetch data from cloud storage (placeholder for implementation)."""
    print("Cloud storage integration not implemented yet")
    return []

def _fetch_from_elasticsearch(config, limit=None):
    """Fetch data from Elasticsearch."""
    try:
        es = elasticsearch.Elasticsearch([config['host']], **config.get('es_config', {}))
        
        query = config.get('query', {"match_all": {}})
        index = config['index']
        
        size = limit if limit else 10000
        
        response = es.search(index=index, body={"query": query, "size": size})
        
        data = []
        for hit in response['hits']['hits']:
            source = hit['_source']
            source['_id'] = hit['_id']  # Add document ID
            data.append(source)
        
        return data
        
    except ImportError:
        print("elasticsearch library not available. Install with: pip install elasticsearch")
        return []
    except Exception as e:
        print(f"Elasticsearch fetch error: {e}")
        return []

def add_data_to_vector_db(
    file_paths,
    db_path='vector_db.index',
    db_type='faiss',
    db_config=None,
    embedding_provider='sentence_transformers',
    embedding_model='all-mpnet-base-v2',
    use_gpu=True,
    use_llama=False,
    api_key=None,
    batch_size=32
):
    """
    Processes multiple files of various formats, extracts content, and adds them to a vector database.
    
    Supported file formats:
    - Documents: PDF, DOC/DOCX, TXT, RTF, LaTeX (.tex)
    - Spreadsheets: XLS/XLSX, CSV
    - Web & Markup: HTML/HTM, Markdown (.md), JSON
    - Presentations: PPT/PPTX
    - eBooks: EPUB, MOBI
    - Media: Images (PNG, JPG, JPEG, TIFF), Videos (MP4, AVI, MOV), Audio (MP3, WAV)

    Args:
        file_paths (list or str): List of paths to the files to process, or single file path.
        db_path (str): Path to the vector database index.
        db_type (str): Type of vector database ('faiss', 'milvus', 'pinecone', 'qdrant', 'weaviate').
        db_config (str): Configuration for the vector database.
        embedding_provider (str): Embedding provider to use.
        embedding_model (str): Embedding model to use.
        use_gpu (bool): Whether to use GPU for embedding.
        use_llama (bool): Whether to use LlamaParse for PDF extraction.
        api_key (str): API key for vector database if required.
        batch_size (int): Size of batches for processing figures and embeddings.
        
    Returns:
        Result from the vector database addition operation.
        
    Raises:
        Exception: If there's an error during file processing or database operations.
    """
    try:
        if isinstance(file_paths, str):
            file_paths = [file_paths]  # Convert single path to list
            
        all_content = []
        all_figures = []
        
        # Validate file paths and extensions
        supported_extensions = {
            '.pdf', '.doc', '.docx', '.txt', '.rtf', '.tex',  # Documents
            '.xls', '.xlsx', '.csv',  # Spreadsheets
            '.html', '.htm', '.md', '.json',  # Web & Markup
            '.ppt', '.pptx',  # Presentations
            '.epub', '.mobi',  # eBooks
            '.png', '.jpg', '.jpeg', '.tiff',  # Images
            '.mp4', '.avi', '.mov',  # Videos
            '.mp3', '.wav'  # Audio
        }
        
        for file_path in file_paths:
            if not os.path.exists(file_path):
                print(f"Warning: File not found: {file_path}")
                continue
                
            _, ext = os.path.splitext(file_path.lower())
            if ext not in supported_extensions:
                print(f"Warning: Unsupported file type '{ext}' for file: {file_path}")
                continue
        
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

        for file_path in file_paths:
            if not os.path.exists(file_path):
                continue  # Already warned above
                
            _, ext = os.path.splitext(file_path.lower())
            if ext not in supported_extensions:
                continue  # Already warned above
                
            print(f"Processing {ext.upper()} file: {file_path}")
            extracted_data = extract_data(file_path, use_llama)
            parsed_texts = extracted_data.get("text", [])
            tables = extracted_data.get("tables", [])
            figures = extracted_data.get("figures", [])

            if not parsed_texts and not tables:
                print(f"No content extracted from file: {file_path}")
                continue

            # Combine text and tables for this file
            file_content = parsed_texts + [str(table) for table in tables]
            all_content.extend(file_content)
            all_figures.extend(figures)

            print(f"Extracted {len(file_content)} items from {file_path}")

        if not all_content:
            print("No content extracted from any files.")
            return

        # Process figures in batches using CLIP
        if all_figures:
            print(f"Processing {len(all_figures)} total figures in batches of {batch_size}")
            for i in range(0, len(all_figures), batch_size):
                batch_figures = all_figures[i:i + batch_size]
                # Process batch of figures
                target_dim = db.get_embedding_dimension()
                captions, clip_embeddings = process_figures_batch(batch_figures, start_idx=i, target_dim=target_dim)
                
                # Add captions to content
                all_content.extend(captions)
                
                # Add embeddings if available
                if clip_embeddings is not None:
                    print(f"Adding batch of {len(clip_embeddings)} figure embeddings")
                    db.add_embeddings(clip_embeddings=clip_embeddings, texts=captions)
                
                print(f"Processed figures {i+1} to {min(i+batch_size, len(all_figures))}")

        # Add all embeddings to the vector database
        result = db.add_embeddings(all_content)

        # Save the FAISS index if applicable
        if db_type == "faiss":
            db.save_index(db_path)
            print(f"FAISS index saved at {db_path}.")
            return result
        else:
            print(f"Data added to {db_type} vector database.")
            return result

    except Exception as e:
        print(f"Error adding files to vector database: {e}")
        raise

# Usage Examples:
if __name__ == "__main__":
    # Example 1: Multiple file types
    mixed_file_paths = [
        'documents/report.pdf',
        'data/spreadsheet.xlsx', 
        'content/article.md',
        'presentations/slides.pptx',
        'data/config.json'
    ]
    
    # Example 2: Single file
    single_file = 'documents/important_doc.docx'
    
    # Example 3: Directory of PDFs (traditional use case)
    pdf_files = ['example1.pdf', 'example2.pdf']
    
    # Example 4: Raw text data
    raw_texts = [
        "This is some content extracted from an API",
        "This is content from a database query",
        "This is content from web scraping"
    ]
    
    # Example 5: Structured data from external source
    structured_data = [
        {"text": "Article content 1", "title": "Article 1", "category": "tech"},
        {"text": "Article content 2", "title": "Article 2", "category": "science"}
    ]
    
    # Process mixed file types
    add_data_to_vector_db(
        mixed_file_paths, 
        db_path='mixed_content_db.index', 
        use_llama=True, 
        batch_size=32
    )
    
    # Add raw text data directly
    add_raw_data_to_vector_db(
        texts=raw_texts,
        db_path='raw_data_db.index'
    )
    
    # Add structured data
    add_structured_data_to_vector_db(
        structured_data,
        metadata_fields=['title', 'category'],
        db_path='structured_db.index'
    )