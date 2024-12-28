from reasonchain.utils.lazy_imports import weaviate, os, psutil
import time
import torch


def connect_to_weaviate_cloud(cluster_url, api_key):
    """
    Connects to a Weaviate Cloud instance.
    """
    if not cluster_url or not api_key:
        raise ValueError("Both cluster_url and api_key must be provided for cloud connection.")
    
    try:
        client = weaviate.Client(
            url=cluster_url,
            auth_client_secret=weaviate.auth.AuthApiKey(api_key),
        )
        # Check connection
        if not client.is_ready():
            raise RuntimeError("Weaviate Cloud is not ready.")
        print("Connected to Weaviate Cloud successfully.")
        return client
    except Exception as e:
        raise RuntimeError(f"Error connecting to Weaviate Cloud: {e}")


class WeaviateVectorDB:
    def __init__(self, mode="local", host="http://localhost:8080", class_name="VectorObject", dimension=768, api_key=None, WEAVIATE_CLUSTER_URL=None):
        """
        Initializes a connection to a Weaviate instance (local or cloud).
        """
        self.class_name = class_name
        self.dimension = dimension
        self.start_time = time.time()
        self.device_type = "GPU" if torch.cuda.is_available() else "CPU"

        # Connect to Weaviate based on mode
        if mode == "cloud":
            self.client = connect_to_weaviate_cloud(
                cluster_url=WEAVIATE_CLUSTER_URL,
                api_key=api_key,
            )
        elif mode == "local":
            self.client = weaviate.Client(url=host)
        else:
            raise ValueError("Mode must be either 'local' or 'cloud'.")

        self._initialize_schema()

    def _initialize_schema(self):
        """
        Initialize the Weaviate schema for the specified class.
        """
        schema = self.client.schema.get()
        if not any(cls["class"] == self.class_name for cls in schema.get("classes", [])):
            self.client.schema.create_class({
                "class": self.class_name,
                "vectorizer": "none",  # Using custom vectors
                "properties": [],
            })
            print(f"Class {self.class_name} created.")
        else:
            print(f"Class {self.class_name} already exists.")

    def add_embeddings(self, ids, embeddings):
        """
        Add embeddings to Weaviate with metrics tracking.
        """
        try:
            embedding_start = time.time()

            for id_, embedding in zip(ids, embeddings):
                if len(embedding) != self.dimension:
                    raise ValueError(f"Embedding dimension mismatch. Expected {self.dimension}, got {len(embedding)}")
                self.client.data_object.create(
                    data_object={"id": id_},
                    class_name=self.class_name,
                    vector=embedding,
                )

            embedding_time = time.time() - embedding_start

            # Resource metrics
            gpu_memory = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            cpu_memory = psutil.Process().memory_info().rss / 1024**2

            print(f"Inserted {len(ids)} embeddings in {embedding_time:.4f} seconds.")
            print(f"GPU memory used: {gpu_memory:.2f} MB, CPU memory used: {cpu_memory:.2f} MB")

        except Exception as e:
            print(f"Error adding embeddings: {e}")

    def search(self, query_embedding, top_k=5):
        """
        Search for the nearest neighbors with metrics tracking.
        """
        try:
            search_start = time.time()

            response = self.client.query.get(
                self.class_name,
                ["id"]
            ).with_near_vector({"vector": query_embedding}).with_limit(top_k).do()

            search_time = time.time() - search_start

            # Resource metrics
            gpu_memory = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            cpu_memory = psutil.Process().memory_info().rss / 1024**2

            matches = response.get("data", {}).get("Get", {}).get(self.class_name, [])
            results = [
                {
                    "id": match["id"],
                    "distance": match["_additional"]["distance"],
                    "metadata": {
                        "search_time": search_time,
                        "query_time": time.time() - self.start_time,
                        "gpu_memory_used": gpu_memory,
                        "cpu_memory_used": cpu_memory,
                        "device_type": self.device_type,
                    },
                }
                for match in matches
            ]

            print(f"Search completed in {search_time:.4f} seconds.")
            print(f"GPU memory used: {gpu_memory:.2f} MB, CPU memory used: {cpu_memory:.2f} MB")
            return results

        except Exception as e:
            print(f"Error during search: {e}")
            return []

    def get_all(self):
        """
        Retrieve all objects and vectors from the Weaviate class.
        """
        try:
            results = []
            response = self.client.query.get(
                self.class_name,
                ["id"]
            ).with_limit(10000).with_additional(["vector"]).do()

            objects = response.get("data", {}).get("Get", {}).get(self.class_name, [])
            for obj in objects:
                results.append({
                    "id": obj["id"],
                    "vector": obj["_additional"]["vector"],
                })
            print(f"Retrieved {len(results)} records from the class {self.class_name}.")
            return results

        except Exception as e:
            raise RuntimeError(f"Error retrieving all records from Weaviate: {e}")
