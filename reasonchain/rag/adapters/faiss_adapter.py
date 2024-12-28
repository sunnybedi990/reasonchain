from reasonchain.utils.lazy_imports import numpy as np, faiss, pickle, os, psutil
import time
import sys
import torch

class FAISSVectorDB:
    def __init__(self, use_gpu=True, dimension=768):
        self.use_gpu = use_gpu
        self.id_map = {}
        self.metadata_map = {}
        self.dimension = dimension
        self.start_time = time.time()
        
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
        """Add embeddings with associated metadata."""
        try:
            embedding_start = time.time()
            start_id = self.index.ntotal
            self.index.add(embeddings)
            embedding_time = time.time() - embedding_start
            
            # Resource usage
            gpu_memory = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            cpu_memory = psutil.Process().memory_info().rss / 1024**2
            
            # Store texts and metadata
            for i, text in enumerate(texts):
                idx = start_id + i
                self.id_map[idx] = text
                
                # Base metadata
                base_metadata = {
                    "text": text,
                    "index": idx,
                    "timestamp": time.time(),
                    "embedding_time": embedding_time,
                    "gpu_memory_used": gpu_memory,
                    "cpu_memory_used": cpu_memory,
                    "device_type": self.device_type,
                    "num_threads": self.num_threads,
                    "vector_dimension": self.dimension,
                    "chunk_size": len(text),
                }
                
                # Merge with provided metadata if exists
                if metadata and i < len(metadata):
                    base_metadata.update(metadata[i])
                
                self.metadata_map[idx] = base_metadata
                
        except Exception as e:
            raise ValueError(f"Error adding embeddings: {e}")

    def search(self, query_embedding, top_k=5):
        """Search with comprehensive metadata."""
        try:
            search_start = time.time()
            distances, indices = self.index.search(np.array([query_embedding]), top_k)
            search_time = time.time() - search_start
            
            # Resource metrics
            gpu_memory = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            cpu_memory = psutil.Process().memory_info().rss / 1024**2
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx in self.id_map:
                    # Get existing metadata and update with search-specific info
                    metadata = self.metadata_map.get(idx, {}).copy()
                    metadata.update({
                        "search_time": search_time,
                        "query_time": time.time() - self.start_time,
                        "similarity_score": float(distances[0][i]),
                        "rank": i + 1,
                        "total_results": len(indices[0]),
                        "gpu_memory_used": gpu_memory,
                        "cpu_memory_used": cpu_memory,
                        "index_size": self.index.ntotal,
                    })
                    
                    results.append({
                        "text": self.id_map[idx],
                        "score": float(distances[0][i]),
                        "metadata": metadata,
                        "index": int(idx)
                    })
            
            # Add global search metadata
            if results:
                global_metadata = {
                    "max_score": float(min(distances[0])),  # Lower distance = higher similarity
                    "min_score": float(max(distances[0])),
                    "mean_score": float(np.mean(distances[0])),
                    "total_chunks": self.index.ntotal,
                    "query_time": search_time,
                }
                for result in results:
                    result["metadata"].update(global_metadata)
                    
            return results
            
        except Exception as e:
            raise ValueError(f"Error during search: {e}")

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

