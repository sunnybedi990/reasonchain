from reasonchain.utils. lazy_imports import os, pinecone, numpy as np, time, psutil, torch
import re

def sanitize_index_name(name):
    """
    Sanitizes the index name to conform to Pinecone's requirements.
    Converts to lowercase, replaces invalid characters with hyphens,
    and ensures the name doesn't start or end with a hyphen.
    """
    sanitized = re.sub(r'[^a-z0-9-]', '-', name.lower())
    sanitized = re.sub(r'^-+|-+$', '', sanitized)  # Remove leading/trailing hyphens
    return sanitized


class PineconeVectorDB:
    def __init__(self, api_key=None, environment="us-east-1", index_name="vector_index",dimension=768):
        if not api_key:
            api_key = os.getenv("PINECONE_API_KEY")
            if not api_key:
                raise ValueError("Pinecone API key not found. Set it in the environment or pass it explicitly.")
        index_name = sanitize_index_name(index_name)
        print(f"Initializing Pinecone index '{index_name}' with dimension: {dimension}")

        # Initialize Pinecone client
        self.pinecone = pinecone.Pinecone(api_key=api_key)

        # Check if the index exists; create if it doesn't
        if index_name not in self.pinecone.list_indexes().names():
            print(f"Creating index '{index_name}'...")
            self.pinecone.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",  # Metric can be 'euclidean', 'cosine', or 'dotproduct'
                spec=pinecone.ServerlessSpec(
                    cloud="aws",  # Specify the cloud provider
                    region=environment  # Specify the environment/region
                )
            )
        print(index_name)
        # Connect to the index
        self.index = self.pinecone.Index(index_name)

    def add_embeddings(self, embeddings, texts, namespace="default-namespace", metadata_key="text"):
        """
        Adds embeddings to the Pinecone index, including metadata.

        Args:
            embeddings (list): A list of embeddings (e.g., vectors).
            texts (list): A list of corresponding texts.
            namespace (str): Namespace for grouping vectors in Pinecone.
            metadata_key (str): Key under which text will be stored as metadata.
        """
        embedding_start = time.time()

        vectors = [
            {
                "id": f"vec-{i}",
                "values": embedding.tolist(),  # Convert numpy array to list
                "metadata": {
                        metadata_key: text,
                        "embedding_time": time.time() - embedding_start,
                        "device_type": self.device_type,
                        "gpu_memory_used": torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
                        "cpu_memory_used": psutil.Process().memory_info().rss / 1024**2,
                    }
            }
            for i, (embedding, text) in enumerate(zip(embeddings, texts))
        ]

        # Use Pinecone's upsert API to insert the vectors
        upsert_response = self.index.upsert(
            vectors=vectors,
            namespace=namespace
        )

        print(f"Pinecone upsert response: {upsert_response}")



    def search(self, query_embedding, top_k=5,namespace="default-namespace"):
        """
        Searches for nearest neighbors in the Pinecone index.
        
        Args:
            query_embedding (list or np.ndarray): The query vector for the search.
            top_k (int): The number of nearest neighbors to retrieve.

        Returns:
            list of tuples: Each tuple contains (text or id, score) from the search results.
        """
        try:
            search_start = time.time()

            # Ensure the query embedding is a list
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()

            # Perform search
            response = self.index.query(
                vector=query_embedding, 
                top_k=top_k, 
                namespace=namespace, 
                include_values=True, 
                include_metadata=True
            )

            search_time = time.time() - search_start
            gpu_memory = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            cpu_memory = psutil.Process().memory_info().rss / 1024**2

            # Extract and enrich results
            results = []
            for match in response.get('matches', []):
                enriched_metadata = match['metadata']
                enriched_metadata.update({
                    "search_time": search_time,
                    "device_type": self.device_type,
                    "gpu_memory_used": gpu_memory,
                    "cpu_memory_used": cpu_memory,
                    "query_time": time.time() - self.start_time,
                    "similarity_score": match['score'],
                })
                results.append({
                    "id": match['id'],
                    "metadata": enriched_metadata,
                    "score": match['score'],
                })

            return results

        except Exception as e:
            raise RuntimeError(f"Error during Pinecone search: {e}")

    def get_all(self, namespace="default-namespace"):
        """
        Retrieve all vectors and metadata from the Pinecone index.
        """
        try:
            # Use a dummy vector to retrieve all items
            query_response = self.index.query(
                vector=[0] * self.dimension,  # Dummy vector
                top_k=10000,  # Adjust based on index size
                namespace=namespace,
                include_metadata=True
            )

            return [
                {
                    "id": match["id"],
                    "metadata": match["metadata"]
                }
                for match in query_response.get("matches", [])
            ]

        except Exception as e:
            raise RuntimeError(f"Error retrieving all records from Pinecone: {e}")
