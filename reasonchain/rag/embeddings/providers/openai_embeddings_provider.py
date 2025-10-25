"""OpenAI Embeddings Provider Implementation"""

import os
from typing import List, Optional
from reasonchain.llm_models.base_provider import BaseEmbeddingProvider
from reasonchain.utils.lazy_imports import openai, dotenv

dotenv.load_dotenv()


class OpenAIEmbeddingsProvider(BaseEmbeddingProvider):
    """
    OpenAI embeddings provider.
    
    Supports OpenAI embedding models:
    - text-embedding-ada-002 (1536d)
    - text-embedding-3-small (1536d)
    - text-embedding-3-large (3072d)
    """
    
    def __init__(self, model_name: str = "text-embedding-ada-002", api_key: Optional[str] = None, **kwargs):
        """
        Initialize OpenAI embeddings provider.
        
        Args:
            model_name (str): OpenAI embedding model name
            api_key (str, optional): OpenAI API key
            **kwargs: Additional configuration
        """
        super().__init__(model_name, api_key, **kwargs)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        try:
            self.client = openai.OpenAI(api_key=self.api_key)
            
            # Set dimensions based on model
            dimension_map = {
                "text-embedding-ada-002": 1536,
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072
            }
            self._dimension = dimension_map.get(model_name, 1536)
            
            print(f"Initialized OpenAI embeddings: {model_name} ({self._dimension}d)")
        except Exception as e:
            print(f"Error initializing OpenAI embeddings: {e}")
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
            response = self.client.embeddings.create(
                input=text,
                model=self.model_name
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating OpenAI embedding: {e}")
            return []
    
    def embed_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts (List[str]): List of texts to embed
            batch_size (int): Batch size (OpenAI supports up to ~2048 texts)
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        try:
            all_embeddings = []
            
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model_name
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            
            return all_embeddings
        except Exception as e:
            print(f"Error generating OpenAI batch embeddings: {e}")
            return []
    
    def get_dimension(self) -> int:
        """
        Get the embedding dimension.
        
        Returns:
            int: Embedding dimension
        """
        return self._dimension

