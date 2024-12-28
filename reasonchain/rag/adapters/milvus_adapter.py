from reasonchain.utils.lazy_imports import os, pymilvus, psutil
import time
import torch


class MilvusVectorDB:
    def __init__(self, collection_name, dimension=768, host="localhost", port="19530"):
        """
        Initialize the Milvus vector database with a unique collection name.
        """
        self.collection_name = collection_name
        self.dimension = dimension
        self.start_time = time.time()
        self.device_type = "GPU" if torch.cuda.is_available() else "CPU"

        # Connect to Milvus
        pymilvus.connections.connect(host=host, port=port)
        self.collection = self._initialize_or_load_collection()

    def _initialize_or_load_collection(self):
        """
        Initialize or load the collection in Milvus.
        """
        fields = [
            pymilvus.FieldSchema(name="id", dtype=pymilvus.DataType.INT64, is_primary=True, auto_id=True),
            pymilvus.FieldSchema(name="embedding", dtype=pymilvus.DataType.FLOAT_VECTOR, dim=self.dimension),
        ]
        schema = pymilvus.CollectionSchema(fields)
        return pymilvus.Collection(name=self.collection_name, schema=schema)

    def add_embeddings(self, embeddings, metadata=None):
        """
        Add embeddings and track metrics like time and resource usage.
        """
        try:
            embedding_start = time.time()

            # Insert embeddings into the collection
            ids = [i for i in range(len(embeddings))]
            self.collection.insert([ids, embeddings])

            embedding_time = time.time() - embedding_start
            gpu_memory = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            cpu_memory = psutil.Process().memory_info().rss / 1024**2

            # Generate metadata for each vector
            metadata_records = []
            for i, embedding in enumerate(embeddings):
                record_metadata = {
                    "index": ids[i],
                    "embedding_time": embedding_time,
                    "gpu_memory_used": gpu_memory,
                    "cpu_memory_used": cpu_memory,
                    "device_type": self.device_type,
                    "vector_dimension": self.dimension,
                }

                if metadata and i < len(metadata):
                    record_metadata.update(metadata[i])

                metadata_records.append(record_metadata)

            print(f"Inserted {len(embeddings)} embeddings with metadata.")
            return metadata_records

        except Exception as e:
            raise RuntimeError(f"Error adding embeddings to Milvus: {e}")

    def search(self, query_embedding, top_k=5):
        """
        Search for nearest neighbors with comprehensive metadata.
        """
        try:
            search_start = time.time()

            # Perform the search
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param={"metric_type": "L2", "params": {"nprobe": 10}},
                limit=top_k,
            )

            search_time = time.time() - search_start
            gpu_memory = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            cpu_memory = psutil.Process().memory_info().rss / 1024**2

            # Extract and enrich results
            enriched_results = []
            for result in results[0]:
                metadata = result.entity.get("metadata", {})
                metadata.update({
                    "search_time": search_time,
                    "query_time": time.time() - self.start_time,
                    "similarity_score": result.distance,
                    "device_type": self.device_type,
                    "gpu_memory_used": gpu_memory,
                    "cpu_memory_used": cpu_memory,
                })

                enriched_results.append({
                    "id": result.id,
                    "embedding": result.entity.get("embedding"),
                    "metadata": metadata,
                    "score": result.distance,
                })

            return enriched_results

        except Exception as e:
            raise RuntimeError(f"Error during Milvus search: {e}")

    def drop_collection(self):
        """
        Drop the current collection.
        """
        try:
            if self.collection:
                print(f"Dropping collection: {self.collection_name}")
                self.collection.drop()
        except Exception as e:
            raise RuntimeError(f"Error dropping collection: {e}")

    def flush_collection(self):
        """
        Flush the current collection to persist data to storage.
        """
        try:
            self.collection.flush()
            print(f"Collection {self.collection_name} has been flushed to disk.")
        except Exception as e:
            raise RuntimeError(f"Error flushing collection: {e}")

    def get_all(self):
        """
        Retrieve all embeddings and associated metadata from the Milvus collection.
        """
        try:
            results = self.collection.query(
                expr="*", 
                output_fields=["id", "embedding"]
            )
            return [{"id": item["id"], "embedding": item["embedding"]} for item in results]
        except Exception as e:
            raise RuntimeError(f"Error retrieving all records from Milvus: {e}")
