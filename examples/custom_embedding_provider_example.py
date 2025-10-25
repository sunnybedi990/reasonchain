#!/usr/bin/env python3
"""
Custom Embedding Provider Example for ReasonChain

This example demonstrates how to:
1. Use built-in embedding providers
2. Create and register custom embedding providers
3. Use embedding providers with RAG pipelines
4. Integrate with vector databases

Author: ReasonChain Team
"""

import os
import sys
import numpy as np
from typing import List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reasonchain.llm_models.provider_registry import EmbeddingProviderRegistry
from reasonchain.llm_models.base_provider import BaseEmbeddingProvider
from reasonchain.rag.vector.add_to_vector_db import add_raw_data_to_vector_db, add_data_to_vector_db
from reasonchain.rag.rag_main import query_vector_db


# ============================================================================
# EXAMPLE 1: Using Built-in Embedding Providers
# ============================================================================

def example_1_builtin_providers():
    """Example 1: Use built-in embedding providers."""
    print("\n" + "="*70)
    print("EXAMPLE 1: USING BUILT-IN EMBEDDING PROVIDERS")
    print("="*70)
    
    print("\nAvailable built-in embedding providers:")
    providers = EmbeddingProviderRegistry.list_providers()
    for provider in providers:
        print(f"  - {provider}")
    
    # Example usage with sentence_transformers (most common)
    print("\n📝 Adding data with Sentence Transformers...")
    texts = [
        "Artificial intelligence is revolutionizing technology.",
        "Machine learning models learn from data patterns.",
        "Deep learning uses neural networks for complex tasks."
    ]
    
    try:
        add_raw_data_to_vector_db(
            texts=texts,
            db_path="st_embeddings.index",
            db_type="faiss",
            embedding_provider="sentence_transformers",
            embedding_model="all-MiniLM-L6-v2"
        )
        print("✓ Data added successfully with Sentence Transformers!")
    except Exception as e:
        print(f"Note: {e}")
    
    # Example with OpenAI embeddings (requires API key)
    print("\n📝 OpenAI Embeddings example...")
    if os.getenv("OPENAI_API_KEY"):
        try:
            add_raw_data_to_vector_db(
                texts=texts,
                db_path="openai_embeddings.index",
                db_type="faiss",
                embedding_provider="openai",
                embedding_model="text-embedding-ada-002"
            )
            print("✓ Data added successfully with OpenAI embeddings!")
        except Exception as e:
            print(f"Note: {e}")
    else:
        print("  Skipping (OPENAI_API_KEY not set)")


# ============================================================================
# EXAMPLE 2: Create Custom Embedding Provider (Cohere)
# ============================================================================

class CohereEmbeddingProvider(BaseEmbeddingProvider):
    """
    Example custom embedding provider for Cohere.
    
    To use: pip install cohere
    """
    
    def __init__(self, model_name: str = "embed-english-v3.0", api_key: Optional[str] = None, **kwargs):
        super().__init__(model_name, api_key, **kwargs)
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        
        if not self.api_key:
            raise ValueError("Cohere API key required. Set COHERE_API_KEY environment variable.")
        
        try:
            import cohere
            self.client = cohere.Client(self.api_key)
            self._dimension = 1024  # Cohere embed-english-v3.0 dimension
            print(f"Initialized Cohere embeddings: {model_name}")
        except ImportError:
            raise ImportError("Install cohere: pip install cohere")
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for single text."""
        try:
            response = self.client.embed(
                texts=[text],
                model=self.model_name,
                input_type="search_document"
            )
            return response.embeddings[0]
        except Exception as e:
            print(f"Error generating Cohere embedding: {e}")
            return []
    
    def embed_batch(self, texts: List[str], batch_size: int = 96) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        try:
            all_embeddings = []
            
            # Process in batches (Cohere supports up to 96 texts per request)
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                response = self.client.embed(
                    texts=batch,
                    model=self.model_name,
                    input_type="search_document"
                )
                all_embeddings.extend(response.embeddings)
            
            return all_embeddings
        except Exception as e:
            print(f"Error generating Cohere batch embeddings: {e}")
            return []
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension


def example_2_custom_cohere_provider():
    """Example 2: Register and use Cohere embedding provider."""
    print("\n" + "="*70)
    print("EXAMPLE 2: CUSTOM EMBEDDING PROVIDER (COHERE)")
    print("="*70)
    
    try:
        # Register the provider
        EmbeddingProviderRegistry.register('cohere', CohereEmbeddingProvider)
        print("✓ Registered Cohere embedding provider")
        
        # You can now use it!
        if os.getenv("COHERE_API_KEY"):
            texts = ["Sample text 1", "Sample text 2"]
            add_raw_data_to_vector_db(
                texts=texts,
                db_path="cohere_embeddings.index",
                embedding_provider="cohere",
                embedding_model="embed-english-v3.0"
            )
            print("✓ Data added with Cohere embeddings!")
        else:
            print("\nCohere provider registered! To use it:")
            print("  1. Set COHERE_API_KEY environment variable")
            print("  2. add_raw_data_to_vector_db(texts, embedding_provider='cohere')")
    except Exception as e:
        print(f"Note: {e}")


# ============================================================================
# EXAMPLE 3: Custom Voyage AI Embedding Provider
# ============================================================================

class VoyageEmbeddingProvider(BaseEmbeddingProvider):
    """
    Voyage AI embedding provider.
    
    To use: pip install voyageai
    """
    
    def __init__(self, model_name: str = "voyage-01", api_key: Optional[str] = None, **kwargs):
        super().__init__(model_name, api_key, **kwargs)
        self.api_key = api_key or os.getenv("VOYAGE_API_KEY")
        
        if not self.api_key:
            raise ValueError("Voyage API key required")
        
        try:
            import voyageai
            self.client = voyageai.Client(api_key=self.api_key)
            
            # Set dimensions based on model
            dimension_map = {
                "voyage-01": 1024,
                "voyage-lite-01": 1024,
                "voyage-2": 1024
            }
            self._dimension = dimension_map.get(model_name, 1024)
            
            print(f"Initialized Voyage AI embeddings: {model_name}")
        except ImportError:
            raise ImportError("Install voyageai: pip install voyageai")
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for single text."""
        try:
            result = self.client.embed([text], model=self.model_name)
            return result.embeddings[0]
        except Exception as e:
            print(f"Error generating Voyage embedding: {e}")
            return []
    
    def embed_batch(self, texts: List[str], batch_size: int = 128) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        try:
            result = self.client.embed(texts, model=self.model_name)
            return result.embeddings
        except Exception as e:
            print(f"Error generating Voyage batch embeddings: {e}")
            return []
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension


def example_3_voyage_provider():
    """Example 3: Register and use Voyage AI embedding provider."""
    print("\n" + "="*70)
    print("EXAMPLE 3: CUSTOM EMBEDDING PROVIDER (VOYAGE AI)")
    print("="*70)
    
    try:
        # Register the provider
        EmbeddingProviderRegistry.register('voyage', VoyageEmbeddingProvider)
        print("✓ Registered Voyage AI embedding provider")
        
        print("\nVoyage AI provider registered! To use it:")
        print("  1. Set VOYAGE_API_KEY environment variable")
        print("  2. add_data_to_vector_db(files, embedding_provider='voyage', embedding_model='voyage-01')")
    except Exception as e:
        print(f"Note: {e}")


# ============================================================================
# EXAMPLE 4: Local Embedding Model Provider
# ============================================================================

class LocalEmbeddingProvider(BaseEmbeddingProvider):
    """
    Example provider for a local embedding service.
    
    This could be a local API endpoint, ONNX model, or any custom service.
    """
    
    def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs):
        super().__init__(model_name, api_key, **kwargs)
        self.endpoint = kwargs.get('endpoint', 'http://localhost:5000')
        self._dimension = kwargs.get('dimension', 768)
        print(f"Connected to local embedding service at {self.endpoint}")
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for single text."""
        try:
            import requests
            response = requests.post(
                f"{self.endpoint}/embed",
                json={"text": text, "model": self.model_name}
            )
            response.raise_for_status()
            return response.json()['embedding']
        except Exception as e:
            print(f"Error with local embedding service: {e}")
            # Return random embedding for demo purposes
            return np.random.rand(self._dimension).tolist()
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        try:
            import requests
            response = requests.post(
                f"{self.endpoint}/embed_batch",
                json={"texts": texts, "model": self.model_name}
            )
            response.raise_for_status()
            return response.json()['embeddings']
        except Exception as e:
            print(f"Error with local embedding service: {e}")
            # Return random embeddings for demo purposes
            return [np.random.rand(self._dimension).tolist() for _ in texts]
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension


def example_4_local_provider():
    """Example 4: Register and use local embedding provider."""
    print("\n" + "="*70)
    print("EXAMPLE 4: LOCAL EMBEDDING SERVICE PROVIDER")
    print("="*70)
    
    try:
        # Register the provider
        EmbeddingProviderRegistry.register('local_embed', LocalEmbeddingProvider)
        print("✓ Registered local embedding provider")
        
        print("\nLocal embedding provider registered! To use it:")
        print("  provider = EmbeddingProviderRegistry.get_provider(")
        print("      name='local_embed',")
        print("      model_name='your-model',")
        print("      endpoint='http://localhost:5000',")
        print("      dimension=768")
        print("  )")
    except Exception as e:
        print(f"Note: {e}")


# ============================================================================
# EXAMPLE 5: Using Providers Directly
# ============================================================================

def example_5_direct_usage():
    """Example 5: Use embedding providers directly."""
    print("\n" + "="*70)
    print("EXAMPLE 5: USING PROVIDERS DIRECTLY")
    print("="*70)
    
    try:
        # Get a provider instance
        provider = EmbeddingProviderRegistry.get_provider(
            name="sentence_transformers",
            model_name="all-MiniLM-L6-v2"
        )
        
        print(f"\n✓ Got provider: {provider.provider_name}")
        print(f"  Model: {provider.model_name}")
        print(f"  Dimension: {provider.get_dimension()}")
        
        # Generate single embedding
        text = "This is a test sentence."
        embedding = provider.embed_text(text)
        print(f"\n✓ Generated single embedding: {len(embedding)} dimensions")
        print(f"  First 5 values: {embedding[:5]}")
        
        # Generate batch embeddings
        texts = [
            "First document about AI.",
            "Second document about machine learning.",
            "Third document about deep learning."
        ]
        embeddings = provider.embed_batch(texts)
        print(f"\n✓ Generated batch embeddings: {len(embeddings)} texts")
        print(f"  Each embedding: {len(embeddings[0])} dimensions")
        
    except Exception as e:
        print(f"Error: {e}")


# ============================================================================
# EXAMPLE 6: List All Available Providers
# ============================================================================

def example_6_list_providers():
    """Example 6: List all registered embedding providers."""
    print("\n" + "="*70)
    print("EXAMPLE 6: LIST ALL EMBEDDING PROVIDERS")
    print("="*70)
    
    providers = EmbeddingProviderRegistry.list_providers()
    
    print(f"\n🔢 Available Embedding Providers ({len(providers)}):")
    for i, provider_name in enumerate(providers, 1):
        print(f"  {i}. {provider_name}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("REASONCHAIN CUSTOM EMBEDDING PROVIDER EXAMPLES")
    print("Demonstrating Extensible Embedding Architecture")
    print("="*70)
    
    try:
        # Example 1: Built-in providers
        example_1_builtin_providers()
        
        # Example 2: Cohere provider
        example_2_custom_cohere_provider()
        
        # Example 3: Voyage AI provider
        example_3_voyage_provider()
        
        # Example 4: Local embedding service
        example_4_local_provider()
        
        # Example 5: Direct provider usage
        example_5_direct_usage()
        
        # Example 6: List all providers
        example_6_list_providers()
        
        print("\n" + "="*70)
        print("✅ ALL EXAMPLES COMPLETED!")
        print("="*70)
        print("\nKey Takeaways:")
        print("1. ReasonChain supports pluggable embedding providers")
        print("2. Built-in providers: Sentence Transformers, OpenAI, HuggingFace")
        print("3. Add custom providers by implementing BaseEmbeddingProvider")
        print("4. Use with any vector database (FAISS, Pinecone, Milvus, etc.)")
        print("5. Full backward compatibility with existing code")
        
        # Cleanup
        print("\n🧹 Cleaning up test files...")
        import os
        for file in ["st_embeddings.index", "openai_embeddings.index", "cohere_embeddings.index"]:
            if os.path.exists(file):
                os.remove(file)
                print(f"  Removed {file}")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

