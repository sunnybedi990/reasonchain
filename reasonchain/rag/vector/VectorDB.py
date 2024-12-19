import os
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from reasonchain.rag.embeddings.embedding_initializer import initialize_embedding_model
from reasonchain.rag.adapters import FAISSVectorDB, MilvusVectorDB, PineconeVectorDB, QdrantVectorDB, WeaviateVectorDB
import yaml
import json
import numpy as np
from reasonchain.rag.vector.utils import extract_name_from_path, pad_embedding


def load_config():
    # Dynamically locate the config.yaml file
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    print(f"Looking for config at: {config_path}")  # Debugging statement
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        return {}
    except yaml.YAMLError as e:
        print(f"Error reading config file: {e}")
        return {}

def initialize_vector_db(db_type, db_config=None, embedding_dimension=None, db_path=None):
    """
    Dynamically initialize a vector database based on configuration and type.
    Args:
        db_type (str): The type of vector database (e.g., "faiss", "milvus").
        db_config (dict or str, optional): Runtime overrides for database configuration.
        embedding_dimension (int, optional): Dimension of the embeddings.
        db_path (str, optional): Path to derive collection or index name.
    Returns:
        Vector database instance.
    """
    # Load defaults from YAML
    config = load_config()
    db_defaults = config["vector_databases"].get(db_type, {})

    # If db_config is None, use db_defaults
    if db_config is None:
        db_config = db_defaults
    else:
        # Deserialize db_config if it's a JSON string
        if isinstance(db_config, str):
            try:
                db_config = json.loads(db_config)
            except json.JSONDecodeError:
                raise ValueError(f"db_config is not a valid JSON string: {db_config}")
        # Merge defaults with runtime overrides
        db_config = {**db_defaults, **db_config}

    # Extract collection or index name from db_path if applicable
    if db_path:
        derived_name = extract_name_from_path(db_path)
        if "collection_name" in db_config:
            db_config["collection_name"] = derived_name
        if "index_name" in db_config:
            db_config["index_name"] = derived_name

    # Initialize the database adapter
    if db_type == "faiss":
        return FAISSVectorDB(use_gpu=db_config.get("use_gpu", False), dimension=embedding_dimension)
    elif db_type == "milvus":
        return MilvusVectorDB(
            host=db_config["host"],
            port=db_config["port"],
            collection_name=db_config["collection_name"],
            dimension=embedding_dimension,
        )
    elif db_type == "pinecone":
        api_key = db_config.get("api_key") or os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("Pinecone API key is required.")
        return PineconeVectorDB(
            api_key=api_key,
            environment=db_config["environment"],
            index_name=db_config["index_name"],
            dimension=embedding_dimension,
        )
    elif db_type == "qdrant":
        api_key = db_config.get("api_key") or os.getenv("QDRANT_API_KEY")
        if not api_key:
            raise ValueError("Qdrant API key is required.")
        
        return QdrantVectorDB(
            mode=db_config["mode"],
            host=db_config["host"],
            port=db_config["port"],
            collection_name=db_config["collection_name"],
            dimension=embedding_dimension,
            api_key=api_key
        )
    elif db_type == "weaviate":
        api_key = db_config.get("api_key") or os.getenv("PINECONE_API_KEY")
        WEAVIATE_CLUSTER_URL = db_config.get("WEAVIATE_CLUSTER_URL") or os.getenv("WEAVIATE_CLUSTER_URL")
        return WeaviateVectorDB(
            mode=db_config["mode"],
            host=db_config["host"],
            class_name=db_config["class_name"],
            dimension=embedding_dimension,
            api_key=api_key,
            WEAVIATE_CLUSTER_URL=WEAVIATE_CLUSTER_URL
        )
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

    

class VectorDB:
    def __init__(self, db_path,db_type, embedding_provider, embedding_model, db_config=None, use_gpu=True, api_key=None, **kwargs):
        """
        Initializes the appropriate vector database backend dynamically
        and sets up the embedding model.
        """
        print(f"Initializing VectorDB with db_type: {db_type}")
        if db_type == "pinecone" and db_path:
            index_name = os.path.splitext(os.path.basename(db_path))[0]
            kwargs["index_name"] = index_name

        # Initialize the embedding model
        self.model, self.dimension, self.iscallable = initialize_embedding_model(embedding_provider, embedding_model, api_key)

        # Ensure the dimension is passed correctly
        if self.dimension:
            kwargs["dimension"] = self.dimension
            #self.db = get_vector_db_adapter(db_type=db_type, **kwargs)
            self.db = initialize_vector_db(db_type, db_config, self.dimension, db_path=db_path)

        else:
            raise ValueError("Dimension not initialized. Check embedding model configuration.")


        self.use_gpu = use_gpu
        

    def _generate_embeddings(self, texts):
        """
        Generates embeddings for the given texts using the initialized model.
        """
        try:
            if not isinstance(texts, list):
                texts = [texts]  # Ensure input is a list

            if self.iscallable:
                # If the model is callable (e.g., HuggingFace pipeline or OpenAI embeddings)
                embeddings = [self.model(text) for text in texts]
            else:
                # If the model uses an encoder method (e.g., SentenceTransformers)
                embeddings = self.model.encode(texts)

            # Ensure embeddings are in the correct numpy format
            return np.array(embeddings, dtype='float32')

        except Exception as e:
            raise ValueError(f"Error generating embedding: {e}")
        
    def process_clip_embedding(self, clip_embedding, dimension):
        """
        Adapts a CLIP embedding to match the dimension of text embeddings.
        - `clip_embedding`: The original CLIP embedding (NumPy array or tensor).
        - `dimension`: Target dimensionality (e.g., 768).
        """
        try:
            # Flatten CLIP embedding if it's multi-dimensional
            if len(clip_embedding.shape) > 1:
                clip_embedding = clip_embedding.flatten()

            # Ensure consistent dimensions
            if clip_embedding.shape[-1] != dimension:
                clip_embedding = pad_embedding(clip_embedding, dimension)

            return np.array(clip_embedding, dtype='float32')
        except Exception as e:
            raise ValueError(f"Error adapting CLIP embedding: {e}")


    def add_embeddings(self, texts, embeddings=None, clip_embeddings=None, batch_size=32, **kwargs):
        """
        Adds embeddings to the vector database.
        If embeddings are not provided, they will be generated internally.
        """
        if embeddings is None:
            embeddings = self._generate_embeddings(texts)
       
        if clip_embeddings is not None:
            # Ensure each CLIP embedding is adapted to match the expected dimension
            clip_embeddings = [
                self.process_clip_embedding(clip_embedding, self.dimension) for clip_embedding in clip_embeddings
            ]
            clip_embeddings = np.array(clip_embeddings, dtype='float32')
            print(clip_embeddings.shape)
            if len(clip_embeddings.shape) == 3:
                clip_embeddings = clip_embeddings.squeeze(axis=1)  # Remove singleton dimension

            # Combine embeddings and clip_embeddings
            if embeddings is None:
                embeddings = clip_embeddings  # Use clip_embeddings if no other embeddings exist
            else:
                embeddings = np.vstack((embeddings, clip_embeddings))  # Combine embeddings
                texts += [f"clip_embedding_{i}" for i in range(len(clip_embeddings))]  # Placeholder metadata for CLIP
    

        # Check backend type and handle accordingly
        if isinstance(self.db, FAISSVectorDB):
            self.db.add_embeddings(embeddings, texts)  # FAISS generates embeddings internally
        elif isinstance(self.db, PineconeVectorDB):
            namespace = kwargs.get("namespace", "default-namespace")  # Default namespace
            metadata_key = kwargs.get("metadata_key", "text")  # Metadata key for storing text
            self.db.add_embeddings(
                embeddings=embeddings,
                texts=texts,
                namespace=namespace,
                metadata_key=metadata_key
            )
        elif isinstance(self.db, MilvusVectorDB):
            ids = [f"text-{i}" for i in range(len(texts))]
            self.db.add_embeddings(ids, embeddings)
        elif isinstance(self.db, QdrantVectorDB):
            ids = list(range(len(texts)))  # Generate integer IDs
            embeddings = embeddings.tolist()  # Ensure embeddings are in list format
            metadata = [{"text": text} for text in texts]  # Use texts as metadata
            self.db.add_embeddings(ids, embeddings, metadata)
        elif isinstance(self.db, WeaviateVectorDB):
            ids = [f"text-{i}" for i in range(len(texts))]
            self.db.add_embeddings(ids, embeddings)
        else:
            raise ValueError(f"Unsupported backend type: {type(self.db)}")

    def search(self, query, top_k=5):
        """
        Searches for the closest matches in the vector database.
        If the database supports searching, the method generates query embeddings
        and passes them to the adapter's search method.
        """
        # Generate query embedding using VectorDB's embedding model
        query_embedding = self._generate_embeddings([query]).squeeze(0)
        print(query_embedding.shape)
        # Check backend type and delegate search operation
        if isinstance(self.db, FAISSVectorDB):
            return self.db.search(query_embedding, top_k)  # FAISS supports direct search with embeddings
        elif isinstance(self.db, PineconeVectorDB):
            return self.db.search(query_embedding, top_k)
        elif isinstance(self.db, MilvusVectorDB):
            return self.db.search(query_embedding, top_k)
        elif isinstance(self.db, QdrantVectorDB):
            return self.db.search(query_embedding, top_k)
        elif isinstance(self.db, WeaviateVectorDB):
            return self.db.search(query_embedding, top_k)
        else:
            raise ValueError(f"Unsupported backend type: {type(self.db)}")

    def save_index(self, path):
        """
        Saves the index to disk (if supported by the backend).
        """
        if hasattr(self.db, 'save_index'):
            self.db.save_index(path)
        else:
            print("Save index not supported for this backend.")

    def load_index(self, path):
        """
        Loads the index from disk (if supported by the backend).
        """
        if isinstance(self.db, PineconeVectorDB):
            print("Pinecone backend detected. Skipping load_index as it is managed.")
            return

        if hasattr(self.db, 'load_index'):
            self.db.load_index(path)
        else:
            print("Load index not supported for this backend.")
            
    def get_all(self):
            """
            Retrieves all embeddings and metadata from the vector database.
            """
            if hasattr(self.db, "get_all"):
                return self.db.get_all()
            else:
                raise NotImplementedError(f"'get_all' is not implemented for {type(self.db)}.")