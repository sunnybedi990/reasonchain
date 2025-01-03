from reasonchain.utils.lazy_imports import numpy as np, faiss, pickle, os, psutil
import time
import sys
import torch
from dataclasses import dataclass

@dataclass
class FAISSMetrics:
    """Tracks FAISS-specific metrics"""
    # Operation Metrics
    upsert_total: int = 0
    upsert_duration_total: float = 0
    query_total: int = 0
    query_duration_total: float = 0
    avg_query_time_ms: float = 0
    avg_upsert_time_ms: float = 0
    total_api_calls: int = 0
    error_count: int = 0
    
    # Index Stats
    total_vectors: int = 0
    index_size: int = 0
    index_dimensions: int = 0
    
    # Resource Usage
    memory_used_bytes: int = 0
    gpu_memory_used: float = 0
    cpu_memory_used: float = 0
    
    # Performance Metrics
    search_qps: float = 0  # Queries per second
    add_qps: float = 0     # Additions per second
    last_operation_time: float = 0

class FAISSVectorDB:
    def __init__(self, use_gpu=True, dimension=768):
        self.use_gpu = use_gpu
        self.id_map = {}
        self.metadata_map = {}
        self.dimension = dimension
        self.start_time = time.time()
        
        # Initialize metrics
        self.metrics = FAISSMetrics()
        
        # System info
        self.device_type = "GPU" if use_gpu and torch.cuda.is_available() else "CPU"
        self.num_threads = faiss.omp_get_max_threads()

        # Initialize FAISS index
        if use_gpu and torch.cuda.is_available():
            self.res = faiss.StandardGpuResources()
            self.index = faiss.GpuIndexFlatL2(self.res, self.dimension)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)

    def add_embeddings(self, embeddings, texts, metadata=None):
        """Add embeddings with standardized metrics."""
        try:
            embedding_start = time.time()
            
            # Update operation metrics
            self.metrics.upsert_total += len(embeddings)
            self.metrics.total_api_calls += 1
            start_id = self.index.ntotal
            
            # Add embeddings
            self.index.add(embeddings)
            
            # Update metrics
            embedding_time = time.time() - embedding_start
            self.metrics.upsert_duration_total += embedding_time
            self.metrics.avg_upsert_time_ms = (self.metrics.avg_upsert_time_ms * (self.metrics.upsert_total - 1) + 
                                             embedding_time * 1000) / self.metrics.upsert_total
            self.metrics.add_qps = len(embeddings) / embedding_time if embedding_time > 0 else 0
            
            # Resource metrics
            gpu_memory = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            cpu_memory = psutil.Process().memory_info().rss / 1024**2
            
            # Store texts and metadata
            for i, text in enumerate(texts):
                idx = start_id + i
                self.id_map[idx] = text
                
                base_metadata = {
                    "text": text,
                    "index": idx,
                    "timestamp": time.time(),
                    "embedding_time": embedding_time,
                    "chunk_size": len(text),
                }
                
                if metadata and i < len(metadata):
                    base_metadata.update(metadata[i])
                
                self.metadata_map[idx] = base_metadata

            return {
                "status": "success",
                "metadata": {
                    "operation_metrics": {
                        "operation": "add_embeddings",
                        "vectors_added": len(embeddings),
                        "start_id": start_id,
                        "end_id": start_id + len(embeddings) - 1,
                        "embedding_time": embedding_time,
                        "timestamp": time.time(),
                        "upsert_total": self.metrics.upsert_total,
                        "upsert_duration_total": self.metrics.upsert_duration_total,
                        "avg_upsert_time_ms": self.metrics.avg_upsert_time_ms,
                        "total_api_calls": self.metrics.total_api_calls
                    },
                    "resource_metrics": {
                        "gpu_memory_used": gpu_memory,
                        "cpu_memory_used": cpu_memory,
                        "device_type": self.device_type,
                        "memory_used_bytes": self.metrics.memory_used_bytes
                    },
                    "index_metrics": {
                        "total_vectors": self.index.ntotal,
                        "dimension": self.dimension,
                        "index_size": self.metrics.index_size
                    },
                    "performance": {
                        "error_count": self.metrics.error_count,
                        "total_operation_time": time.time() - self.start_time,
                        "add_qps": self.metrics.add_qps,
                        "last_operation_time": embedding_time
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

            distances, indices = self.index.search(np.array([query_embedding]), top_k)
            
            search_time = time.time() - search_start
            self.metrics.query_duration_total += search_time
            self.metrics.avg_query_time_ms = (self.metrics.avg_query_time_ms * (self.metrics.query_total - 1) + 
                                            search_time * 1000) / self.metrics.query_total
            self.metrics.search_qps = 1 / search_time if search_time > 0 else 0

            # Resource metrics
            gpu_memory = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            cpu_memory = psutil.Process().memory_info().rss / 1024**2

            processed_results = []
            scores = distances[0]

            for i, idx in enumerate(indices[0]):
                if idx in self.id_map:
                    processed_results.append({
                        "text": self.id_map[idx],
                        "score": float(distances[0][i]),
                        "metadata": {
                            "search_metrics": {
                                "search_time": search_time,
                                "query_time": time.time() - self.start_time,
                                "similarity_score": float(distances[0][i]),
                                "rank": i + 1,
                                "total_results": len(indices[0]),
                                "query_total": self.metrics.query_total,
                                "query_duration_total": self.metrics.query_duration_total,
                                "avg_query_time_ms": self.metrics.avg_query_time_ms,
                                "total_api_calls": self.metrics.total_api_calls
                            },
                            "resource_metrics": {
                                "gpu_memory_used": gpu_memory,
                                "cpu_memory_used": cpu_memory,
                                "device_type": self.device_type,
                                "memory_used_bytes": self.metrics.memory_used_bytes
                            },
                            "index_metrics": {
                                "total_vectors": self.index.ntotal,
                                "dimension": self.dimension,
                                "index_size": self.metrics.index_size
                            },
                            "score_stats": {
                                "max_score": float(min(distances[0])),  # Lower distance = higher similarity
                                "min_score": float(max(distances[0])),
                                "mean_score": float(np.mean(distances[0])),
                                "total_chunks": self.index.ntotal
                            },
                            "performance": {
                                "search_qps": self.metrics.search_qps,
                                "error_count": self.metrics.error_count,
                                "last_operation_time": search_time
                            }
                        },
                        "index": int(idx)
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

    def save_index(self, path):
        """Save index with metadata."""
        try:
            id_map_path = os.path.splitext(path)[0] + ".pkl"
            metadata_path = os.path.splitext(path)[0] + "_metadata.pkl"

            if self.use_gpu:
                cpu_index = faiss.index_gpu_to_cpu(self.index)
                faiss.write_index(cpu_index, path)
            else:
                faiss.write_index(self.index, path)

            # Save both id_map and metadata_map
            with open(id_map_path, 'wb') as f:
                pickle.dump(self.id_map, f)
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.metadata_map, f)

        except Exception as e:
            raise RuntimeError(f"Error saving FAISS index and metadata: {e}")

    def load_index(self, path):
        """Load index with metadata."""
        try:
            id_map_path = os.path.splitext(path)[0] + ".pkl"
            metadata_path = os.path.splitext(path)[0] + "_metadata.pkl"

            cpu_index = faiss.read_index(path)
            if self.use_gpu:
                self.index = faiss.index_cpu_to_gpu(self.res, 0, cpu_index)
            else:
                self.index = cpu_index

            # Load both id_map and metadata_map
            with open(id_map_path, 'rb') as f:
                self.id_map = pickle.load(f)
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    self.metadata_map = pickle.load(f)

        except Exception as e:
            raise RuntimeError(f"Error loading FAISS index and metadata: {e}")

    def update_id_map(self, texts):
        start_id = self.index.ntotal - len(texts)
        for i, text in enumerate(texts):
            self.id_map[start_id + i] = text
    
    def get_all(self):
        """
        Retrieve all embeddings and their associated texts from the FAISS index.
        """
        total_vectors = self.index.ntotal
        if total_vectors == 0:
            return []
        embeddings = [self.index.reconstruct(i) for i in range(total_vectors)]
        results = [self.id_map[idx] for idx in total_vectors if idx in self.id_map]
        return results

