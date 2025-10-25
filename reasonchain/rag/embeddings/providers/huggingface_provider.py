"""HuggingFace Transformers Embedding Provider Implementation"""

import torch
import numpy as np
from typing import List, Optional
from reasonchain.llm_models.base_provider import BaseEmbeddingProvider
from reasonchain.utils.lazy_imports import transformers


class HuggingFaceProvider(BaseEmbeddingProvider):
    """
    HuggingFace Transformers embedding provider.
    
    Supports any HuggingFace model that can generate embeddings:
    - BERT models (bert-base-uncased, etc.)
    - RoBERTa models
    - DistilBERT models
    - Custom fine-tuned models
    """
    
    def __init__(self, model_name: str = "bert-base-uncased", api_key: Optional[str] = None, **kwargs):
        """
        Initialize HuggingFace provider.
        
        Args:
            model_name (str): HuggingFace model name or path
            api_key (str, optional): HuggingFace token (for private models)
            **kwargs: Additional configuration (device, pooling_strategy, etc.)
        """
        super().__init__(model_name, api_key, **kwargs)
        
        try:
            device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
            self.pooling_strategy = kwargs.get('pooling_strategy', 'mean')  # mean, cls, max
            
            # Load model and tokenizer
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_name,
                token=api_key
            )
            self.model = transformers.AutoModel.from_pretrained(
                model_name,
                token=api_key
            ).to(device)
            
            self.device = device
            self.model.eval()  # Set to evaluation mode
            
            # Get dimension from model config
            self._dimension = self.model.config.hidden_size
            
            print(f"Loaded HuggingFace model: {model_name} ({self._dimension}d) on {device}")
        except Exception as e:
            print(f"Error loading HuggingFace model: {e}")
            raise
    
    def _pool_embeddings(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Pool token embeddings into a single sentence embedding.
        
        Args:
            hidden_states: Token embeddings
            attention_mask: Attention mask
            
        Returns:
            Pooled embeddings
        """
        if self.pooling_strategy == 'cls':
            # Use [CLS] token
            return hidden_states[:, 0]
        elif self.pooling_strategy == 'max':
            # Max pooling
            return torch.max(hidden_states, dim=1)[0]
        else:  # mean pooling (default)
            # Mean pooling with attention mask
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text (str): Text to embed
            
        Returns:
            List[float]: Embedding vector
        """
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                hidden_states = outputs.last_hidden_state
                
                # Pool to get sentence embedding
                embedding = self._pool_embeddings(hidden_states, inputs['attention_mask'])
            
            return embedding.cpu().numpy()[0].tolist()
        except Exception as e:
            print(f"Error generating HuggingFace embedding: {e}")
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
            all_embeddings = []
            
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                # Tokenize batch
                inputs = self.tokenizer(
                    batch,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                # Generate embeddings
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    hidden_states = outputs.last_hidden_state
                    
                    # Pool to get sentence embeddings
                    embeddings = self._pool_embeddings(hidden_states, inputs['attention_mask'])
                
                all_embeddings.extend(embeddings.cpu().numpy().tolist())
            
            return all_embeddings
        except Exception as e:
            print(f"Error generating HuggingFace batch embeddings: {e}")
            return []
    
    def get_dimension(self) -> int:
        """
        Get the embedding dimension.
        
        Returns:
            int: Embedding dimension
        """
        return self._dimension

