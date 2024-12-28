from reasonchain.utils.lazy_imports import os, qdrant_client, psutil
import time
import torch


class QdrantVectorDB:

    def __init__(self, collection_name="vector_collection", dimension=768, mode="local", host="localhost", port=6333, path=None, api_key=None):
        """
        Initialize Qdrant client and collection.
        """
        self.collection_name = collection_name
        self.dimension = dimension
        self.start_time = time.time()
        self.device_type = "GPU" if torch.cuda.is_available() else "CPU"

        if mode == "local":
            self.client = qdrant_client.QdrantClient(host=host, port=port)
        elif mode == "cloud":
            if not api_key:
                raise ValueError("API key is required for cloud mode.")
            cluster_url = os.getenv('QDRANT_CLUSTER_URL')
            self.client = qdrant_client.QdrantClient(url=cluster_url, api_key=api_key)
        elif mode == "memory":
            self.client = qdrant_client.QdrantClient(":memory:")
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        self._initialize_collection()

    def _initialize_collection(self):
        """
        Initialize the Qdrant collection. If the collection already exists, connect to it.
        """
        try:
            existing_collections = [col.name for col in self.client.get_collections().collections]
            if self.collection_name not in existing_collections:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qdrant_client.models.VectorParams(size=self.dimension, distance=qdrant_client.models.Distance.COSINE),
                )
                print(f"Collection '{self.collection_name}' created.")
            else:
                print(f"Collection '{self.collection_name}' already exists. Connected to the collection.")
        except Exception as e:
            raise ValueError(f"Error initializing or connecting to collection: {e}")

    def add_embeddings(self, embeddings, metadata=None):
        """
        Add embeddings to the Qdrant collection with metrics tracking.
        """
        try:
            embedding_start = time.time()
            ids = list(range(len(embeddings)))

            if metadata is None:
                metadata = [{}] * len(ids)

            points = [
                qdrant_client.models.PointStruct(id=ids[i], vector=embeddings[i], payload=metadata[i])
                for i in range(len(ids))
            ]
            self.client.upsert(collection_name=self.collection_name, points=points)
            embedding_time = time.time() - embedding_start

            # Resource metrics
            gpu_memory = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            cpu_memory = psutil.Process().memory_info().rss / 1024**2

            print(f"Inserted {len(embeddings)} embeddings in {embedding_time:.4f} seconds.")
            print(f"GPU memory used: {gpu_memory:.2f} MB, CPU memory used: {cpu_memory:.2f} MB")

        except Exception as e:
            raise RuntimeError(f"Error adding embeddings to Qdrant: {e}")

    def search(self, query_embedding, top_k=5):
        """
        Search for nearest neighbors with metrics tracking.
        """
        try:
            search_start = time.time()

            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                with_payload=True,
            )

            search_time = time.time() - search_start

            # Resource metrics
            gpu_memory = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            cpu_memory = psutil.Process().memory_info().rss / 1024**2

            enriched_results = [
                {
                    "id": point.id,
                    "score": point.score,
                    "payload": point.payload,
                    "metadata": {
                        "search_time": search_time,
                        "query_time": time.time() - self.start_time,
                        "gpu_memory_used": gpu_memory,
                        "cpu_memory_used": cpu_memory,
                        "device_type": self.device_type,
                    },
                }
                for point in results
            ]

            print(f"Search completed in {search_time:.4f} seconds.")
            print(f"GPU memory used: {gpu_memory:.2f} MB, CPU memory used: {cpu_memory:.2f} MB")
            return enriched_results

        except Exception as e:
            raise RuntimeError(f"Error during Qdrant search: {e}")

    def get_all(self):
        """
        Retrieve all embeddings and metadata from the Qdrant collection.
        """
        try:
            results = []
            scroll_id = None
            while True:
                response = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=None,
                    limit=1000,
                    with_payload=True,
                    offset=scroll_id
                )
                results.extend(response.points)
                scroll_id = response.offset
                if not scroll_id:
                    break
            print(f"Retrieved {len(results)} records from the Qdrant collection.")
            return results
        except Exception as e:
            raise RuntimeError(f"Error retrieving all records from Qdrant: {e}")
