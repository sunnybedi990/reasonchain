"""Sentence Transformers Embedding Provider Implementation"""

import numpy as np
from typing import List, Optional
from reasonchain.llm_models.base_provider import BaseEmbeddingProvider
from reasonchain.utils.lazy_imports import sentence_transformers


class SentenceTransformersProvider(BaseEmbeddingProvider):
    """
    Sentence Transformers embedding provider.
    
    Supports all models from sentence-transformers library including:
    - all-mpnet-base-v2 (768d)
    - all-MiniLM-L6-v2 (384d)
    - paraphrase-multilingual-MiniLM-L12-v2 (384d)
    - And any custom sentence-transformers model
    """
    
    def __init__(self, model_name: str = "all-mpnet-base-v2", api_key: Optional[str] = None, **kwargs):
        """
        Initialize Sentence Transformers provider.
        
        Args:
            model_name (str): Model name from sentence-transformers
            api_key (str, optional): Not used for sentence-transformers
            **kwargs: Additional configuration (device, normalize_embeddings, etc.)
        """
        super().__init__(model_name, api_key, **kwargs)
        
        try:
            device = kwargs.get('device', None)
            self.model = sentence_transformers.SentenceTransformer(model_name, device=device)
            self._dimension = self.model.get_sentence_embedding_dimension()
            self.normalize_embeddings = kwargs.get('normalize_embeddings', False)
            print(f"Loaded Sentence Transformer model: {model_name} ({self._dimension}d)")
        except Exception as e:
            print(f"Error loading Sentence Transformer model: {e}")
            raise
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text (str): Text to embed
            
        Returns:
            List[float]: Embedding vector
        """
        try:
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings
            )
            return embedding.tolist()
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return []
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts (List[str]): List of texts to embed
            batch_size (int): Batch size for processing
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings,
                show_progress_bar=len(texts) > 100
            )
            return embeddings.tolist()
        except Exception as e:
            print(f"Error generating batch embeddings: {e}")
            return []
    
    def get_dimension(self) -> int:
        """
        Get the embedding dimension.
        
        Returns:
            int: Embedding dimension
        """
        return self._dimension

