from reasonchain.utils.lazy_imports import weaviate, os, psutil
import time
import torch
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List


@dataclass
class WeaviateMetrics:
    """Tracks Weaviate-specific metrics"""
    # Batch metrics
    batch_durations_ms: Dict[str, List[float]] = field(default_factory=lambda: {
        "object": [],
        "inverted": [],
        "vector": []
    })
    
    # API metrics
    total_api_calls: int = 0
    
    # Memory metrics
    memory_active_bytes: int = 0
    memory_allocated_bytes: int = 0
    
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
        
        # Use specific imports to avoid circular dependency
        auth_credentials = weaviate.classes.init.Auth.api_key(api_key)
        
        # Initialize client with v4 syntax for synchronous cloud connection
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=cluster_url,
            auth_credentials=auth_credentials
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
    def __init__(self, mode="local", host="http://localhost", port=8080, class_name="VectorObject", dimension=768, api_key=None, WEAVIATE_CLUSTER_URL=None):
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
            client = weaviate.connect_to_local(
                host=host,
                port=port,
            )
            self.client = client
        else:
            raise ValueError("Mode must be either 'local' or 'cloud'.")

        self._initialize_schema()

    def _initialize_schema(self):
        """Initialize the Weaviate schema for storing custom vectors."""
        try:
            # Check if collection exists
            if self.client.collections.exists(self.class_name):
                print(f"Using existing collection: {self.class_name}")
                collection = self.client.collections.get(self.class_name)
                
                # Get collection configuration
                collection_config = self.client.collections.export_config(self.class_name)
                
                # Validate the existing collection has required properties
                existing_properties = {
                    prop.name for prop in collection_config.properties
                }
                required_properties = {"doc_id", "text"}
                
                if not required_properties.issubset(existing_properties):
                    missing_props = required_properties - existing_properties
                    raise ValueError(
                        f"Existing collection {self.class_name} missing required properties: {missing_props}"
                    )
                    
                return collection

            # Define properties for new collection
            properties = [
                weaviate.classes.config.Property(
                    name="doc_id",
                    description="Unique identifier for the document",
                    data_type=weaviate.classes.config.DataType.TEXT,
                    skip_vectorization=True,
                    index_filterable=True,
                    index_searchable=True,
                ),
                weaviate.classes.config.Property(
                    name="text",
                    description="The document text content",
                    data_type=weaviate.classes.config.DataType.TEXT,
                    skip_vectorization=True,
                    index_filterable=True,
                    index_searchable=True,
                ),
            ]

            # Create new collection with proper configuration
            collection = self.client.collections.create(
                name=self.class_name,
                description="Collection for storing documents with custom vectors",
                properties=properties,
                vector_index_config=weaviate.classes.config.Configure.VectorIndex.flat(
                    distance_metric=weaviate.classes.config.VectorDistances.COSINE,
                ),
                vectorizer_config=weaviate.classes.config.Configure.Vectorizer.none()  # Important for custom vectors
            )
            
            print(f"Created new collection: {self.class_name}")
            return collection

        except Exception as e:
            print(f"Error initializing schema: {str(e)}")
            raise

    def add_embeddings(self, ids, embeddings, metadata=None):
        """Add embeddings with standardized metrics."""
        try:
            embedding_start = time.time()

            # Update operation metrics
            self.metrics.requests_total["success"] += 1
            self.metrics.total_api_calls += 1

            # Ensure client is connected
            if not self.client.is_ready():
                self.client.connect()

            # Prepare batch of objects
            objects = []
            for id_, embedding, meta in zip(ids, embeddings, metadata):
                if len(embedding) != self.dimension:
                    raise ValueError(f"Embedding dimension mismatch. Expected {self.dimension}, got {len(embedding)}")
                
                # Convert embedding to list if it's a numpy array or tensor
                if hasattr(embedding, 'tolist'):
                    embedding = embedding.tolist()

                # Prepare properties with required fields
                properties = {
                    "doc_id": str(id_),
                    "text": meta["text"]  # Get text from metadata
                }

                objects.append({
                    "properties": properties,
                    "vector": embedding,
                    "class_name": self.class_name,
                })

            # Perform batch import
            failed_objects = []
            with self.client.batch.dynamic() as batch:
                for obj in objects:
                    try:
                        batch.add_object(
                            properties=obj["properties"],
                            vector=obj["vector"],
                            collection=self.class_name,
                        )
                    except Exception as batch_error:
                        failed_objects.append({"object": obj, "error": str(batch_error)})
                        self.metrics.requests_total["failed"] += 1

            # Update metrics for successful adds
            successful_adds = len(objects) - len(failed_objects)
            self.metrics.vector_index_operations["insert"] += successful_adds
            self.metrics.vector_index_size += successful_adds
            embedding_time = time.time() - embedding_start
            print(f"Batch insertion completed in {embedding_time:.2f} seconds.")
# Resource metrics
            gpu_memory = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            cpu_memory = psutil.Process().memory_info().rss / 1024**2
            return {
                "status": "success" if not failed_objects else "partial_success",
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
            self.metrics.total_api_calls += 1

            # Ensure client is connected
            if not self.client.is_ready():
                self.client.connect()

            # Get collection
            collection = self.client.collections.get(self.class_name)
            
            # Convert query_embedding to list if it's numpy array or tensor
            if hasattr(query_embedding, 'tolist'):
                query_embedding = query_embedding.tolist()
            
            # Perform search with proper v4 syntax
            response = collection.query.near_vector(
                near_vector=query_embedding,
                limit=top_k,
                return_properties=["text", "doc_id"],  # Add doc_id to returned properties
                return_metadata=weaviate.classes.query.MetadataQuery(
                    distance=True,
                    score=True,
                )
            )

            search_time = time.time() - search_start
            processed_results = []
            scores = []
            
            # Process results with proper error handling
            if hasattr(response, 'objects') and response.objects:
                for i, obj in enumerate(response.objects):
                    # Get distance and convert to similarity score
                    distance = obj.metadata.distance if hasattr(obj.metadata, 'distance') else 0.0
                    score = 1.0 - (distance / 2.0)  # Convert cosine distance to similarity
                    scores.append(score)
                    
                    # Extract text content
                    text_content = obj.properties.get("text", "")
                    
                    processed_results.append({
                        "text": text_content,
                        "score": score,
                        "metadata": {
                            "search_metrics": {
                                "search_time": search_time,
                                "query_time": time.time() - self.start_time,
                                "similarity_score": score,
                                "rank": i + 1,
                                "total_results": len(response.objects),
                                "total_api_calls": self.metrics.total_api_calls,
                                "requests_total": self.metrics.requests_total
                            },
                            "resource_metrics": {
                                "gpu_memory_used": torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
                                "cpu_memory_used": psutil.Process().memory_info().rss / 1024**2,
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
                        "index": obj.properties.get("doc_id", str(obj.uuid) if hasattr(obj, 'uuid') else None)
                    })

            if not processed_results:
                print("Warning: No results found in search response")
                print(f"Debug - Response content: {response}")
                
            return processed_results

        except Exception as e:
            self.metrics.requests_total["failed"] += 1
            print(f"Search error: {str(e)}")
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


