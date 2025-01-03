from reasonchain.utils.lazy_imports import os, qdrant_client, psutil
import time
import torch
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List

@dataclass
class QdrantMetrics:
    """Tracks Qdrant-specific metrics"""
    # Operation Metrics
    upsert_total: int = 0
    upsert_duration_total: float = 0
    query_total: int = 0
    query_duration_total: float = 0
    avg_query_time_ms: float = 0
    avg_upsert_time_ms: float = 0
    total_api_calls: int = 0
    error_count: int = 0
    
    # Collection Stats
    total_vectors: int = 0
    index_size: int = 0
    index_dimensions: int = 0
    index_fullness: float = 0
    
    # Resource Usage
    write_unit_total: int = 0
    read_unit_total: int = 0
    storage_size_bytes: int = 0
    
    # Performance Metrics
    grpc_responses_total: int = 0
    rest_responses_total: int = 0
    grpc_responses_duration: List[float] = field(default_factory=list)
    rest_responses_duration: List[float] = field(default_factory=list)

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
        self.metrics = QdrantMetrics()

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

    def add_embeddings(self, ids, embeddings, metadata=None):
        """Add embeddings with standardized metrics."""
        try:
            embedding_start = time.time()
            
            # Update operation metrics
            self.metrics.upsert_total += len(embeddings)
            self.metrics.total_api_calls += 1
            
            # Create points
            points = [
                qdrant_client.models.PointStruct(
                    id=ids[i], 
                    vector=embeddings[i], 
                    payload=metadata[i] if metadata else {}
                )
                for i in range(len(ids))
            ]

            # Perform upsert
            operation = self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

            # Update metrics
            embedding_time = time.time() - embedding_start
            self.metrics.upsert_duration_total += embedding_time
            self.metrics.avg_upsert_time_ms = (self.metrics.avg_upsert_time_ms * (self.metrics.upsert_total - 1) + 
                                             embedding_time * 1000) / self.metrics.upsert_total
            
            # Resource metrics
            gpu_memory = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            cpu_memory = psutil.Process().memory_info().rss / 1024**2

            return {
                "status": "success",
                "metadata": {
                    "operation_metrics": {
                        "operation": "add_embeddings",
                        "vectors_added": len(points),
                        "start_id": ids[0],
                        "end_id": ids[-1],
                        "embedding_time": embedding_time,
                        "timestamp": time.time(),
                        "upsert_total": self.metrics.upsert_total,
                        "upsert_duration_total": self.metrics.upsert_duration_total,
                        "avg_upsert_time_ms": self.metrics.avg_upsert_time_ms,
                        "total_api_calls": self.metrics.total_api_calls
                    },
                    "resource_metrics": {
                        "write_unit_total": self.metrics.write_unit_total,
                        "storage_size_bytes": self.metrics.storage_size_bytes,
                        "gpu_memory_used": gpu_memory,
                        "cpu_memory_used": cpu_memory,
                        "device_type": self.device_type
                    },
                    "index_metrics": {
                        "total_vectors": self.collection_info.vectors_count,
                        "dimension": self.dimension,
                        "index_fullness": self.metrics.index_fullness
                    },
                    "performance": {
                        "error_count": self.metrics.error_count,
                        "total_operation_time": time.time() - self.start_time,
                        "grpc_responses_total": self.metrics.grpc_responses_total,
                        "rest_responses_total": self.metrics.rest_responses_total
                    }
                }
            }

        except Exception as e:
            self.metrics.error_count += 1
            return {
                "status": "error",
                "metadata": {
                    "operation_metrics": {
                        "operation": "add_embeddings",
                        "error": str(e),
                        "timestamp": time.time(),
                        "attempted_vectors": len(embeddings),
                        "error_count": self.metrics.error_count,
                        "total_api_calls": self.metrics.total_api_calls
                    }
                }
            }

    def search(self, query_embedding, top_k=5):
        """Search with standardized metrics."""
        try:
            search_start = time.time()
            
            # Update operation metrics
            self.metrics.query_total += 1
            self.metrics.total_api_calls += 1

            # Get collection info first
            collection_info = self.client.get_collection(self.collection_name)
            vectors_count = collection_info.points_count

            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                with_payload=True,
            )

            search_time = time.time() - search_start
            self.metrics.query_duration_total += search_time
            self.metrics.avg_query_time_ms = (self.metrics.avg_query_time_ms * (self.metrics.query_total - 1) + 
                                            search_time * 1000) / self.metrics.query_total

            # Resource metrics
            gpu_memory = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            cpu_memory = psutil.Process().memory_info().rss / 1024**2

            processed_results = []
            scores = [point.score for point in results]

            for i, point in enumerate(results):
                processed_results.append({
                    "text": point.payload.get("text", ""),
                    "score": point.score,
                    "metadata": {
                        "search_metrics": {
                            "search_time": search_time,
                            "query_time": time.time() - self.start_time,
                            "similarity_score": point.score,
                            "rank": i + 1,
                            "total_results": len(results),
                            "query_total": self.metrics.query_total,
                            "query_duration_total": self.metrics.query_duration_total,
                            "avg_query_time_ms": self.metrics.avg_query_time_ms,
                            "total_api_calls": self.metrics.total_api_calls
                        },
                        "resource_metrics": {
                            "read_unit_total": self.metrics.read_unit_total,
                            "gpu_memory_used": gpu_memory,
                            "cpu_memory_used": cpu_memory,
                            "device_type": self.device_type
                        },
                        "index_metrics": {
                            "total_vectors": vectors_count,
                            "dimension": self.dimension,
                            "index_fullness": self.metrics.index_fullness
                        },
                        "score_stats": {
                            "max_score": max(scores),
                            "min_score": min(scores),
                            "mean_score": sum(scores) / len(scores),
                            "total_chunks": vectors_count
                        },
                        "performance": {
                            "grpc_responses_total": self.metrics.grpc_responses_total,
                            "rest_responses_total": self.metrics.rest_responses_total,
                            "error_count": self.metrics.error_count
                        }
                    },
                    "index": point.id
                })

            return processed_results

        except Exception as e:
            self.metrics.error_count += 1
            raise ValueError(f"Error during search: {e}")

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

    def get_metadata(self) -> Dict[str, Any]:
        """Get comprehensive metadata about the collection and operations."""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            
            return {
                # Collection Information
                "collection_name": self.collection_name,
                "vectors_count": collection_info.vectors_count,
                "dimension": self.dimension,
                "distance": self.distance,
                
                # Operation Metrics
                "rest_responses": {
                    "total": self.metrics.rest_responses_total,
                    "failed": self.metrics.rest_responses_fail_total,
                    "avg_duration": sum(self.metrics.rest_responses_duration) / len(self.metrics.rest_responses_duration) if self.metrics.rest_responses_duration else 0,
                    "min_duration": min(self.metrics.rest_responses_duration) if self.metrics.rest_responses_duration else 0,
                    "max_duration": max(self.metrics.rest_responses_duration) if self.metrics.rest_responses_duration else 0,
                },
                "grpc_responses": {
                    "total": self.metrics.grpc_responses_total,
                    "failed": self.metrics.grpc_responses_fail_total,
                    "avg_duration": sum(self.metrics.grpc_responses_duration) / len(self.metrics.grpc_responses_duration) if self.metrics.grpc_responses_duration else 0,
                    "min_duration": min(self.metrics.grpc_responses_duration) if self.metrics.grpc_responses_duration else 0,
                    "max_duration": max(self.metrics.grpc_responses_duration) if self.metrics.grpc_responses_duration else 0,
                },
                
                # Memory Metrics
                "memory": {
                    "active_bytes": self.metrics.memory_active_bytes,
                    "allocated_bytes": self.metrics.memory_allocated_bytes,
                    "metadata_bytes": self.metrics.memory_metadata_bytes,
                    "resident_bytes": self.metrics.memory_resident_bytes,
                    "retained_bytes": self.metrics.memory_retained_bytes,
                },
                
                # Collection Metrics
                "collection": {
                    "pending_operations": self.metrics.collection_pending_operations,
                    "grpc_requests": self.metrics.collection_grpc_requests,
                    "rest_requests": self.metrics.collection_rest_requests,
                },
                
                # Resource Usage
                "resources": {
                    "cpu_usage": self.metrics.cpu_usage,
                    "disk_usage_bytes": self.metrics.disk_usage_bytes,
                    "network_receive_bytes": self.metrics.network_receive_bytes,
                    "network_transmit_bytes": self.metrics.network_transmit_bytes,
                    "gpu_memory_used": torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
                    "cpu_memory_used": psutil.Process().memory_info().rss / 1024**2,
                },
                
                # System Info
                "system": {
                    "timestamp": datetime.now().isoformat(),
                    "uptime": time.time() - self.start_time,
                }
            }
        except Exception as e:
            print(f"Error getting metadata: {e}")
            return {}

    def _update_resource_metrics(self):
        """Update resource usage metrics."""
        try:
            process = psutil.Process()
            self.metrics.cpu_usage = process.cpu_percent()
            memory_info = process.memory_info()
            self.metrics.memory_resident_bytes = memory_info.rss
            self.metrics.memory_allocated_bytes = memory_info.vms
            
            # Update network metrics if available
            net_io = psutil.net_io_counters()
            self.metrics.network_receive_bytes = net_io.bytes_recv
            self.metrics.network_transmit_bytes = net_io.bytes_sent
            
            # Update disk metrics
            disk_io = psutil.disk_io_counters()
            self.metrics.disk_usage_bytes = disk_io.read_bytes + disk_io.write_bytes
            
        except Exception as e:
            print(f"Error updating resource metrics: {e}")
