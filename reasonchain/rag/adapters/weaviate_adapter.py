from reasonchain.utils.lazy_imports import os, psutil
import time
import torch
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List
import weaviate

@dataclass
class WeaviateMetrics:
    """Tracks Weaviate-specific metrics"""
    # Batch metrics
    batch_durations_ms: Dict[str, List[float]] = field(default_factory=lambda: {
        "object": [],
        "inverted": [],
        "vector": []
    })
    
    # Object metrics
    object_count: int = 0
    objects_durations_ms: Dict[str, List[float]] = field(default_factory=dict)
    
    # Vector index metrics
    vector_index_tombstones: int = 0
    vector_index_size: int = 0
    vector_index_operations: Dict[str, int] = field(default_factory=lambda: {
        "insert": 0,
        "delete": 0,
        "update": 0
    })
    
    # Queue metrics
    index_queue_size: int = 0
    index_queue_metrics: Dict[str, float] = field(default_factory=lambda: {
        "push_duration_ms": 0,
        "delete_duration_ms": 0,
        "search_duration_ms": 0,
        "preload_duration_ms": 0,
        "wait_duration_ms": 0
    })
    
    # Request metrics
    requests_total: Dict[str, int] = field(default_factory=lambda: {
        "success": 0,
        "failed": 0
    })
    
    # Performance metrics
    startup_durations_ms: List[float] = field(default_factory=list)
    maintenance_durations_ms: List[float] = field(default_factory=list)

def connect_to_weaviate_cloud(cluster_url, api_key):
    """
    Connects to a Weaviate Cloud instance using v4 client.
    """
    if not cluster_url or not api_key:
        raise ValueError("Both cluster_url and api_key must be provided for cloud connection.")
    
    try:
        # Clean up the URL
        cluster_url = cluster_url.strip()
        if cluster_url.endswith('/'):
            cluster_url = cluster_url[:-1]
        
        # Ensure proper URL format
        if not cluster_url.startswith('https://'):
            if cluster_url.startswith('http://'):
                cluster_url = 'https://' + cluster_url[7:]
            else:
                cluster_url = f'https://{cluster_url}'
        
        print(f"Debug - Final URL: {cluster_url}")
        print(f"Debug - API Key (first 10 chars): {api_key[:10]}...")
        
        # Initialize client with v4 syntax for synchronous cloud connection
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=cluster_url,
            auth_credentials=weaviate.classes.init.Auth.api_key(api_key)
        )
        
        print(f"Debug - Attempting connection to: {cluster_url}")
        
        # Test connection
        if client.is_ready():
            print("Connected to Weaviate Cloud successfully.")
            return client
        else:
            raise RuntimeError("Client reports not ready")
                
    except Exception as e:
        print(f"Debug - Detailed connection error: {str(e)}")
        print(f"Debug - Error type: {type(e)}")
        raise RuntimeError(f"Error connecting to Weaviate Cloud: {e}")


class WeaviateVectorDB:
    def __init__(self, mode="local", host="http://localhost:8080", class_name="VectorObject", dimension=768, api_key=None, WEAVIATE_CLUSTER_URL=None):
        """Initialize Weaviate client and collection with metrics tracking."""
        self.class_name = class_name
        self.dimension = dimension
        self.start_time = time.time()
        self.device_type = "GPU" if torch.cuda.is_available() else "CPU"
        self.metrics = WeaviateMetrics()

        # Connect to Weaviate based on mode
        if mode == "cloud":
            if not WEAVIATE_CLUSTER_URL:
                WEAVIATE_CLUSTER_URL = os.getenv("WEAVIATE_CLUSTER_URL")
            if not api_key:
                api_key = os.getenv("WEAVIATE_API_KEY")
                
            if not WEAVIATE_CLUSTER_URL or not api_key:
                raise ValueError("For cloud mode, both WEAVIATE_CLUSTER_URL and WEAVIATE_API_KEY must be provided")
                
            print(f"Connecting to Weaviate Cloud at: {WEAVIATE_CLUSTER_URL}")
            self.client = connect_to_weaviate_cloud(
                cluster_url=WEAVIATE_CLUSTER_URL,
                api_key=api_key,
            )
        elif mode == "local":
            # For local connection
            client = weaviate.connect_to_weaviate(
                connection_params=weaviate.connect.ConnectionParams.from_url(
                    url=host
                )
            )
            self.client = client
        else:
            raise ValueError("Mode must be either 'local' or 'cloud'.")

        self._initialize_schema()

    def _initialize_schema(self):
        """
        Initialize the Weaviate schema for the specified class.
        """
        try:
            schema = self.client.schema.get()
            if not any(cls["class"] == self.class_name for cls in schema.get("classes", [])):
                self.client.schema.create_class({
                    "class": self.class_name,
                    "vectorizer": "none",
                    "properties": [],
                })
                print(f"Class {self.class_name} created.")
            else:
                print(f"Class {self.class_name} already exists.")
        except Exception as e:
            raise ValueError(f"Error initializing schema: {e}")

    def add_embeddings(self, ids, embeddings):
        """Add embeddings with standardized metrics."""
        try:
            embedding_start = time.time()
            
            # Update operation metrics
            self.metrics.requests_total["success"] += 1
            self.metrics.vector_index_operations["insert"] += len(embeddings)
            self.metrics.total_api_calls += 1

            for id_, embedding in zip(ids, embeddings):
                if len(embedding) != self.dimension:
                    raise ValueError(f"Embedding dimension mismatch. Expected {self.dimension}, got {len(embedding)}")
                self.client.data_object.create(
                    data_object={"id": id_},
                    class_name=self.class_name,
                    vector=embedding,
                )

            embedding_time = time.time() - embedding_start
            self.metrics.batch_durations_ms["vector"].append(embedding_time * 1000)
            self.metrics.index_queue_metrics["push_duration_ms"] = embedding_time * 1000

            # Resource metrics
            gpu_memory = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            cpu_memory = psutil.Process().memory_info().rss / 1024**2

            return {
                "status": "success",
                "metadata": {
                    "operation_metrics": {
                        "operation": "add_embeddings",
                        "vectors_added": len(embeddings),
                        "start_id": ids[0],
                        "end_id": ids[-1],
                        "embedding_time": embedding_time,
                        "timestamp": time.time(),
                        "batch_metrics": self.metrics.batch_durations_ms,
                        "total_api_calls": self.metrics.total_api_calls,
                        "requests_total": self.metrics.requests_total
                    },
                    "resource_metrics": {
                        "gpu_memory_used": gpu_memory,
                        "cpu_memory_used": cpu_memory,
                        "device_type": self.device_type,
                        "memory_active_bytes": self.metrics.memory_active_bytes,
                        "memory_allocated_bytes": self.metrics.memory_allocated_bytes
                    },
                    "index_metrics": {
                        "total_vectors": self.metrics.vector_index_size,
                        "dimension": self.dimension,
                        "vector_operations": self.metrics.vector_index_operations,
                        "tombstones": self.metrics.vector_index_tombstones
                    },
                    "queue_metrics": {
                        "size": self.metrics.index_queue_size,
                        **self.metrics.index_queue_metrics
                    },
                    "performance": {
                        "startup_duration_avg": sum(self.metrics.startup_durations_ms)/len(self.metrics.startup_durations_ms) if self.metrics.startup_durations_ms else 0,
                        "maintenance_duration_avg": sum(self.metrics.maintenance_durations_ms)/len(self.metrics.maintenance_durations_ms) if self.metrics.maintenance_durations_ms else 0,
                        "error_count": self.metrics.requests_total["failed"]
                    }
                }
            }

        except Exception as e:
            self.metrics.requests_total["failed"] += 1
            return {
                "status": "error",
                "metadata": {
                    "operation_metrics": {
                        "operation": "add_embeddings",
                        "error": str(e),
                        "timestamp": time.time(),
                        "attempted_vectors": len(embeddings),
                        "error_count": self.metrics.requests_total["failed"],
                        "total_api_calls": self.metrics.total_api_calls
                    }
                }
            }

    def search(self, query_embedding, top_k=5):
        """Search with standardized metrics."""
        try:
            search_start = time.time()
            self.metrics.requests_total["success"] += 1
            self.metrics.total_api_calls += 1

            response = self.client.query.get(
                self.class_name,
                ["id"]
            ).with_near_vector({"vector": query_embedding}).with_limit(top_k).do()

            search_time = time.time() - search_start
            self.metrics.index_queue_metrics["search_duration_ms"] = search_time * 1000

            # Resource metrics
            gpu_memory = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            cpu_memory = psutil.Process().memory_info().rss / 1024**2

            matches = response.get("data", {}).get("Get", {}).get(self.class_name, [])
            scores = [match["_additional"]["distance"] for match in matches]

            processed_results = []
            for i, match in enumerate(matches):
                processed_results.append({
                    "text": match.get("text", ""),
                    "score": match["_additional"]["distance"],
                    "metadata": {
                        "search_metrics": {
                            "search_time": search_time,
                            "query_time": time.time() - self.start_time,
                            "similarity_score": match["_additional"]["distance"],
                            "rank": i + 1,
                            "total_results": len(matches),
                            "total_api_calls": self.metrics.total_api_calls,
                            "requests_total": self.metrics.requests_total
                        },
                        "resource_metrics": {
                            "gpu_memory_used": gpu_memory,
                            "cpu_memory_used": cpu_memory,
                            "device_type": self.device_type,
                            "memory_active_bytes": self.metrics.memory_active_bytes,
                            "memory_allocated_bytes": self.metrics.memory_allocated_bytes
                        },
                        "index_metrics": {
                            "total_vectors": self.metrics.vector_index_size,
                            "dimension": self.dimension,
                            "vector_operations": self.metrics.vector_index_operations,
                            "tombstones": self.metrics.vector_index_tombstones
                        },
                        "score_stats": {
                            "max_score": max(scores),
                            "min_score": min(scores),
                            "mean_score": sum(scores) / len(scores),
                            "total_chunks": self.metrics.vector_index_size
                        },
                        "queue_metrics": {
                            "size": self.metrics.index_queue_size,
                            **self.metrics.index_queue_metrics
                        }
                    },
                    "index": match["id"]
                })

            return processed_results

        except Exception as e:
            self.metrics.requests_total["failed"] += 1
            raise ValueError({
                "error": str(e),
                "metadata": {
                    "operation": "search",
                    "error_count": self.metrics.requests_total["failed"],
                    "total_api_calls": self.metrics.total_api_calls,
                    "timestamp": time.time()
                }
            })

    def get_metadata(self) -> Dict[str, Any]:
        """Get comprehensive metadata about operations and metrics."""
        try:
            return {
                # Batch Operations
                "batch_metrics": {
                    "durations_ms": {
                        op: {
                            "avg": sum(durations)/len(durations) if durations else 0,
                            "max": max(durations) if durations else 0,
                            "min": min(durations) if durations else 0
                        }
                        for op, durations in self.metrics.batch_durations_ms.items()
                    }
                },
                
                # Vector Index Stats
                "vector_index": {
                    "tombstones": self.metrics.vector_index_tombstones,
                    "size": self.metrics.vector_index_size,
                    "operations": self.metrics.vector_index_operations
                },
                
                # Queue Metrics
                "index_queue": {
                    "size": self.metrics.index_queue_size,
                    **self.metrics.index_queue_metrics
                },
                
                # Request Stats
                "requests": self.metrics.requests_total,
                
                # Performance Metrics
                "performance": {
                    "startup_duration_avg": sum(self.metrics.startup_durations_ms)/len(self.metrics.startup_durations_ms) if self.metrics.startup_durations_ms else 0,
                    "maintenance_duration_avg": sum(self.metrics.maintenance_durations_ms)/len(self.metrics.maintenance_durations_ms) if self.metrics.maintenance_durations_ms else 0
                },
                
                # Resource Usage
                "resources": {
                    "gpu_memory_used": torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
                    "cpu_memory_used": psutil.Process().memory_info().rss / 1024**2,
                    "device_type": self.device_type
                },
                
                # System Info
                "system": {
                    "timestamp": datetime.now().isoformat(),
                    "uptime": time.time() - self.start_time
                }
            }
        except Exception as e:
            print(f"Error getting metadata: {e}")
            return {}

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


def main():
    client = WeaviateVectorDB(mode="cloud", api_key=os.getenv("WEAVIATE_API_KEY"), WEAVIATE_CLUSTER_URL=os.getenv("WEAVIATE_CLUSTER_URL"))
    print(client.is_ready())

if __name__ == "__main__":
    main()