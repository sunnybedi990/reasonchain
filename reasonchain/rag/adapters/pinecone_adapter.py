from reasonchain.utils. lazy_imports import os, pinecone, numpy as np, time, psutil, torch
import re
from dataclasses import dataclass
from typing import Dict, Any, List
from datetime import datetime

def sanitize_index_name(name):
    """
    Sanitizes the index name to conform to Pinecone's requirements:
    - Only lowercase alphanumeric characters and hyphens allowed
    - Cannot start or end with a hyphen
    """
    # Convert to lowercase and replace underscores with hyphens
    sanitized = name.lower().replace('_', '-')
    
    # Replace any other invalid characters with hyphens
    sanitized = re.sub(r'[^a-z0-9-]', '-', sanitized)
    
    # Remove leading/trailing hyphens
    sanitized = re.sub(r'^-+|-+$', '', sanitized)
    
    return sanitized

@dataclass
class PineconeMetrics:
    """Tracks Pinecone-specific metrics"""
    # Database Records
    record_total: int = 0
    
    # Operation Metrics
    upsert_total: int = 0
    upsert_duration_total: float = 0
    query_total: int = 0
    query_duration_total: float = 0
    fetch_total: int = 0
    fetch_duration_total: float = 0
    update_total: int = 0
    update_duration_total: float = 0
    delete_total: int = 0
    delete_duration_total: float = 0
    
    # Resource Usage
    write_unit_total: int = 0
    read_unit_total: int = 0
    storage_size_bytes: int = 0
    
    # Index Stats
    index_size: int = 0
    index_dimensions: int = 0
    index_fullness: float = 0
    
    # Cloud Metrics
    cloud_provider: str = "unknown"
    region: str = "unknown"
    replicas: int = 1
    
    # Performance Metrics
    avg_query_time_ms: float = 0
    avg_upsert_time_ms: float = 0
    total_api_calls: int = 0
    error_count: int = 0

class PineconeVectorDB:
    def __init__(self, api_key=None, environment="us-east-1", index_name="vector_index", dimension=768, batch_size=1000):
        if not api_key:
            api_key = os.getenv("PINECONE_API_KEY")
            if not api_key:
                raise ValueError("Pinecone API key not found. Set it in the environment or pass it explicitly.")
        
        # Sanitize the index name first
        self.index_name = sanitize_index_name(index_name)
        self.dimension = dimension
        self.start_time = time.time()
        self.device_type = "GPU" if torch.cuda.is_available() else "CPU"
        self.metrics = PineconeMetrics()
        self.batch_size = batch_size
        self.environment = environment
        # Initialize Pinecone client
        self.pinecone = pinecone.Pinecone(api_key=api_key)

        # Check if the index exists; create if it doesn't
        if self.index_name not in self.pinecone.list_indexes().names():
            print(f"Creating index '{self.index_name}'...")
            self.pinecone.create_index(
                name=self.index_name,
                dimension=dimension,
                metric="cosine",
                spec=pinecone.ServerlessSpec(
                    cloud="aws",
                    region=self.environment
                )
            )

        # Connect to the index
        self.index = self.pinecone.Index(self.index_name)
        
        # Initialize metrics
        self._update_index_stats()

    def _update_index_stats(self):
        """Update index statistics."""
        try:
            # Get index stats
            stats = self.index.describe_index_stats()
            self.metrics.record_total = stats.total_vector_count
            self.metrics.index_size = stats.total_vector_count
            self.metrics.index_dimensions = self.dimension
            self.metrics.storage_size_bytes = stats.total_vector_count * self.dimension * 4  # Approximate size
            
            # Get index description (using describe_index_stats instead of describe_index)
            index_desc = stats.namespaces
            self.metrics.cloud_provider = "aws"  # Default for serverless
            self.metrics.region = self.environment  # Get region from index object
            self.metrics.replicas = 1  # Default for serverless
            
        except Exception as e:
            print(f"Error updating index stats: {e}")

    def get_metadata(self) -> Dict[str, Any]:
        """Get comprehensive metadata about the index and operations."""
        try:
            index_info = self.index.describe_index_stats()
            index_desc = self.index.describe_index()
            
            return {
                # Index Information
                "index_name": self.index_name,
                "cloud": self.metrics.cloud_provider,
                "region": self.metrics.region,
                "capacity_mode": index_desc.get("capacity_mode", "unknown"),
                "dimension": self.dimension,
                
                # Operation Metrics
                "operation_stats": {
                    "record_total": self.metrics.record_total,
                    "upsert_total": self.metrics.upsert_total,
                    "upsert_duration_total": self.metrics.upsert_duration_total,
                    "query_total": self.metrics.query_total,
                    "query_duration_total": self.metrics.query_duration_total,
                    "fetch_total": self.metrics.fetch_total,
                    "fetch_duration_total": self.metrics.fetch_duration_total,
                    "update_total": self.metrics.update_total,
                    "update_duration_total": self.metrics.update_duration_total,
                    "delete_total": self.metrics.delete_total,
                    "delete_duration_total": self.metrics.delete_duration_total,
                },
                
                # Resource Usage
                "resource_usage": {
                    "write_unit_total": self.metrics.write_unit_total,
                    "read_unit_total": self.metrics.read_unit_total,
                    "storage_size_bytes": self.metrics.storage_size_bytes,
                    "cpu_memory_used": psutil.Process().memory_info().rss / 1024**2,
                    "gpu_memory_used": torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
                },
                
                # Performance Metrics
                "performance": {
                    "avg_query_time_ms": self.metrics.avg_query_time_ms,
                    "avg_upsert_time_ms": self.metrics.avg_upsert_time_ms,
                    "total_api_calls": self.metrics.total_api_calls,
                    "error_count": self.metrics.error_count,
                    "total_operation_time": time.time() - self.start_time,
                },
                
                # Index Stats
                "index_stats": {
                    "total_vectors": self.metrics.index_size,
                    "dimension": self.metrics.index_dimensions,
                    "index_fullness": self.metrics.index_fullness,
                    "replicas": self.metrics.replicas,
                },
                
                # Cloud Metrics
                "cloud_metrics": {
                    "provider": self.metrics.cloud_provider,
                    "region": self.metrics.region,
                    "replicas": self.metrics.replicas,
                },
                
                # System Info
                "system": {
                    "timestamp": datetime.now().isoformat(),
                    "device_type": self.device_type,
                    "uptime": time.time() - self.start_time,
                }
            }
        except Exception as e:
            print(f"Error getting metadata: {e}")
            return {}

    def add_embeddings(self, embeddings, texts, namespace="default-namespace", metadata_key="text"):
        """Add embeddings with comprehensive metrics."""
        try:
            embedding_start = time.time()
            
            # Update operation metrics
            self.metrics.upsert_total += len(embeddings)
            self.metrics.total_api_calls += 1
            
            vectors = [
                {
                    "id": f"vec-{i}",
                    "values": embedding.tolist(),
                    "metadata": {
                        metadata_key: text,
                        "timestamp": time.time(),
                    }
                }
                for i, (embedding, text) in enumerate(zip(embeddings, texts))
            ]

            # Add debug print
            print(f"Adding {len(embeddings)} vectors to index '{self.index_name}'")
            
            # Perform upsert
            for i in range(0, len(vectors), self.batch_size):
                batch = vectors[i:i + self.batch_size]
                upsert_response = self.index.upsert(
                    vectors=batch,
                    namespace=namespace
                )
            
            # Add verification print
            stats = self.index.describe_index_stats()
            print(f"Total vectors after upsert: {stats.total_vector_count}")

            # Update metrics
            embedding_time = time.time() - embedding_start
            self.metrics.upsert_duration_total += embedding_time
            self.metrics.avg_upsert_time_ms = (self.metrics.avg_upsert_time_ms * (self.metrics.upsert_total - 1) + 
                                             embedding_time * 1000) / self.metrics.upsert_total
            self._update_index_stats()

            return {
                "status": "success",
                "metadata": {
                    "operation_metrics": {
                        "operation": "add_embeddings",
                        "vectors_added": len(vectors),
                        "start_id": f"vec-0",
                        "end_id": f"vec-{len(vectors)-1}",
                        "embedding_time": embedding_time,
                        "timestamp": time.time(),
                        "namespace": namespace,
                        "upsert_total": self.metrics.upsert_total,
                        "upsert_duration_total": self.metrics.upsert_duration_total,
                        "avg_upsert_time_ms": self.metrics.avg_upsert_time_ms,
                        "total_api_calls": self.metrics.total_api_calls
                    },
                    "resource_metrics": {
                        "write_unit_total": self.metrics.write_unit_total,
                        "storage_size_bytes": self.metrics.storage_size_bytes,
                        "gpu_memory_used": torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
                        "cpu_memory_used": psutil.Process().memory_info().rss / 1024**2,
                        "device_type": self.device_type
                    },
                    "index_metrics": {
                        "total_vectors": self.metrics.index_size,
                        "dimension": self.metrics.index_dimensions,
                        "index_fullness": self.metrics.index_fullness,
                    },
                    "cloud_metrics": {
                        "provider": self.metrics.cloud_provider,
                        "region": self.metrics.region,
                        "replicas": self.metrics.replicas,
                    },
                    "performance": {
                        "error_count": self.metrics.error_count,
                        "total_operation_time": time.time() - self.start_time,
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

    def search(self, query_embedding, top_k=5, namespace="default-namespace"):
        """Search with comprehensive metrics."""
        try:
            search_start = time.time()
            
            # Update operation metrics
            self.metrics.query_total += 1
            self.metrics.total_api_calls += 1

            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()

            response = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=namespace,
                include_values=True,
                include_metadata=True
            )

            matches = response.get('matches', [])
            
            if not matches:
                print(f"No matches found. Total vectors in index: {self.metrics.index_size}")
                return []

            search_time = time.time() - search_start
            self.metrics.query_duration_total += search_time
            self.metrics.avg_query_time_ms = (self.metrics.avg_query_time_ms * (self.metrics.query_total - 1) + 
                                            search_time * 1000) / self.metrics.query_total
            
            scores = [match['score'] for match in matches]
            
            processed_results = []
            for i, match in enumerate(matches):
                processed_results.append({
                    "text": match['metadata'].get('text', ''),
                    "score": match['score'],
                    "metadata": {
                        "search_metrics": {
                            "search_time": search_time,
                            "query_time": time.time() - self.start_time,
                            "similarity_score": match['score'],
                            "rank": i + 1,
                            "total_results": len(matches),
                            "query_total": self.metrics.query_total,
                            "query_duration_total": self.metrics.query_duration_total,
                            "avg_query_time_ms": self.metrics.avg_query_time_ms,
                            "total_api_calls": self.metrics.total_api_calls
                        },
                        "resource_metrics": {
                            "read_unit_total": self.metrics.read_unit_total,
                            "gpu_memory_used": torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
                            "cpu_memory_used": psutil.Process().memory_info().rss / 1024**2,
                            "device_type": self.device_type
                        },
                        "index_metrics": {
                            "total_vectors": self.metrics.index_size,
                            "dimension": self.metrics.index_dimensions,
                            "index_fullness": self.metrics.index_fullness,
                        },
                        "score_stats": {
                            "max_score": max(scores),
                            "min_score": min(scores),
                            "mean_score": sum(scores) / len(scores),
                            "total_chunks": self.metrics.index_size
                        },
                        "cloud_metrics": {
                            "provider": self.metrics.cloud_provider,
                            "region": self.metrics.region,
                            "replicas": self.metrics.replicas,
                        }
                    },
                    "index": match['id']
                })

            return processed_results

        except Exception as e:
            self.metrics.error_count += 1
            raise ValueError({
                "error": str(e),
                "metadata": {
                    "operation": "search",
                    "error_count": self.metrics.error_count,
                    "total_api_calls": self.metrics.total_api_calls,
                    "timestamp": time.time()
                }
            })

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
