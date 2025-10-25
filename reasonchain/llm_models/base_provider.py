"""
Base LLM Provider Interface

This module defines the abstract base class for LLM providers, enabling
a plugin architecture where users can easily add support for new LLM services.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    All LLM providers must implement this interface to be compatible with ReasonChain.
    This enables users to easily add support for new LLM services (Anthropic, Cohere,
    AI21, local models, etc.) without modifying core code.
    """
    
    def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs):
        """
        Initialize the LLM provider.
        
        Args:
            model_name (str): Name/identifier of the model
            api_key (str, optional): API key for the service
            **kwargs: Additional provider-specific configuration
        """
        self.model_name = model_name
        self.api_key = api_key
        self.config = kwargs
    
    @abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt (str): Input prompt
            **kwargs: Additional generation parameters (temperature, max_tokens, etc.)
            
        Returns:
            str: Generated response
        """
        pass
    
    @abstractmethod
    def generate_chat_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate a response using chat format.
        
        Args:
            messages (List[Dict]): List of message dicts with 'role' and 'content'
            **kwargs: Additional generation parameters
            
        Returns:
            str: Generated response
        """
        pass
    
    def summarize(self, text: str, max_tokens: int = 150, **kwargs) -> str:
        """
        Summarize the given text.
        
        Args:
            text (str): Text to summarize
            max_tokens (int): Maximum tokens for summary
            **kwargs: Additional parameters
            
        Returns:
            str: Summarized text
        """
        prompt = f"Summarize the following text concisely:\n\n{text}"
        return self.generate_response(prompt, max_tokens=max_tokens, **kwargs)
    
    def list_models(self) -> List[str]:
        """
        List available models for this provider.
        
        Returns:
            List[str]: Available model names
        """
        return [self.model_name]
    
    def validate_config(self) -> bool:
        """
        Validate the provider configuration.
        
        Returns:
            bool: True if configuration is valid
        """
        return True
    
    @property
    def provider_name(self) -> str:
        """Return the name of this provider."""
        return self.__class__.__name__.replace('Provider', '').lower()


class BaseEmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers.
    
    All embedding providers must implement this interface to be compatible with ReasonChain.
    """
    
    def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs):
        """
        Initialize the embedding provider.
        
        Args:
            model_name (str): Name/identifier of the embedding model
            api_key (str, optional): API key for the service
            **kwargs: Additional provider-specific configuration
        """
        self.model_name = model_name
        self.api_key = api_key
        self.config = kwargs
        self._dimension = None
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text (str): Text to embed
            
        Returns:
            List[float]: Embedding vector
        """
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts (List[str]): List of texts to embed
            batch_size (int): Batch size for processing
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """
        Get the embedding dimension.
        
        Returns:
            int: Embedding dimension
        """
        pass
    
    @property
    def provider_name(self) -> str:
        """Return the name of this provider."""
        return self.__class__.__name__.replace('Provider', '').lower()


class ProviderConfig:
    """Configuration class for provider settings."""
    
    def __init__(self, 
                 temperature: float = 0.7,
                 max_tokens: int = 2000,
                 top_p: float = 1.0,
                 frequency_penalty: float = 0.0,
                 presence_penalty: float = 0.0,
                 **kwargs):
        """
        Initialize provider configuration.
        
        Args:
            temperature (float): Sampling temperature
            max_tokens (int): Maximum tokens to generate
            top_p (float): Nucleus sampling parameter
            frequency_penalty (float): Frequency penalty
            presence_penalty (float): Presence penalty
            **kwargs: Additional custom parameters
        """
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.custom_params = kwargs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'top_p': self.top_p,
            'frequency_penalty': self.frequency_penalty,
            'presence_penalty': self.presence_penalty,
            **self.custom_params
        }

