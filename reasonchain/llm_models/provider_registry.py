"""
LLM Provider Registry

This module manages the registration and retrieval of LLM providers,
enabling a plugin architecture for ReasonChain.
"""

import os
from typing import Dict, Type, Optional, Any
from reasonchain.llm_models.base_provider import BaseLLMProvider, BaseEmbeddingProvider


class LLMProviderRegistry:
    """
    Registry for LLM providers.
    
    Allows users to register custom LLM providers and retrieve them by name.
    This enables a plugin architecture where new LLM services can be added
    without modifying core code.
    """
    
    _providers: Dict[str, Type[BaseLLMProvider]] = {}
    _instances: Dict[str, BaseLLMProvider] = {}
    
    @classmethod
    def register(cls, name: str, provider_class: Type[BaseLLMProvider]):
        """
        Register a new LLM provider.
        
        Args:
            name (str): Provider name (e.g., 'openai', 'anthropic', 'cohere')
            provider_class (Type[BaseLLMProvider]): Provider class implementing BaseLLMProvider
            
        Example:
            >>> LLMProviderRegistry.register('anthropic', AnthropicProvider)
        """
        if not issubclass(provider_class, BaseLLMProvider):
            raise ValueError(f"{provider_class} must inherit from BaseLLMProvider")
        
        cls._providers[name.lower()] = provider_class
        print(f"Registered LLM provider: {name}")
    
    @classmethod
    def get_provider(cls, 
                     name: str, 
                     model_name: str, 
                     api_key: Optional[str] = None,
                     **kwargs) -> BaseLLMProvider:
        """
        Get an instance of a registered provider.
        
        Args:
            name (str): Provider name
            model_name (str): Model name
            api_key (str, optional): API key
            **kwargs: Additional provider configuration
            
        Returns:
            BaseLLMProvider: Provider instance
            
        Raises:
            ValueError: If provider not found
        """
        name = name.lower()
        if name not in cls._providers:
            raise ValueError(
                f"Provider '{name}' not found. Available providers: {list(cls._providers.keys())}"
            )
        
        # Create cache key
        cache_key = f"{name}:{model_name}"
        
        # Return cached instance if available
        if cache_key in cls._instances:
            return cls._instances[cache_key]
        
        # Create new instance
        provider_class = cls._providers[name]
        instance = provider_class(model_name=model_name, api_key=api_key, **kwargs)
        cls._instances[cache_key] = instance
        
        return instance
    
    @classmethod
    def list_providers(cls) -> list:
        """
        List all registered providers.
        
        Returns:
            list: List of provider names
        """
        return list(cls._providers.keys())
    
    @classmethod
    def unregister(cls, name: str) -> bool:
        """
        Unregister a provider.
        
        Args:
            name (str): Provider name
            
        Returns:
            bool: True if provider was unregistered
        """
        name = name.lower()
        if name in cls._providers:
            del cls._providers[name]
            # Clear cached instances
            cls._instances = {k: v for k, v in cls._instances.items() if not k.startswith(f"{name}:")}
            print(f"Unregistered LLM provider: {name}")
            return True
        return False
    
    @classmethod
    def clear_cache(cls):
        """Clear all cached provider instances."""
        cls._instances = {}


class EmbeddingProviderRegistry:
    """
    Registry for embedding providers.
    
    Similar to LLMProviderRegistry but for embedding models.
    """
    
    _providers: Dict[str, Type[BaseEmbeddingProvider]] = {}
    _instances: Dict[str, BaseEmbeddingProvider] = {}
    
    @classmethod
    def register(cls, name: str, provider_class: Type[BaseEmbeddingProvider]):
        """
        Register a new embedding provider.
        
        Args:
            name (str): Provider name (e.g., 'openai_embeddings', 'cohere_embeddings')
            provider_class (Type[BaseEmbeddingProvider]): Provider class
            
        Example:
            >>> EmbeddingProviderRegistry.register('voyage', VoyageEmbeddingProvider)
        """
        if not issubclass(provider_class, BaseEmbeddingProvider):
            raise ValueError(f"{provider_class} must inherit from BaseEmbeddingProvider")
        
        cls._providers[name.lower()] = provider_class
        print(f"Registered embedding provider: {name}")
    
    @classmethod
    def get_provider(cls,
                     name: str,
                     model_name: str,
                     api_key: Optional[str] = None,
                     **kwargs) -> BaseEmbeddingProvider:
        """
        Get an instance of a registered embedding provider.
        
        Args:
            name (str): Provider name
            model_name (str): Model name
            api_key (str, optional): API key
            **kwargs: Additional configuration
            
        Returns:
            BaseEmbeddingProvider: Provider instance
        """
        name = name.lower()
        if name not in cls._providers:
            raise ValueError(
                f"Embedding provider '{name}' not found. Available: {list(cls._providers.keys())}"
            )
        
        cache_key = f"{name}:{model_name}"
        
        if cache_key in cls._instances:
            return cls._instances[cache_key]
        
        provider_class = cls._providers[name]
        instance = provider_class(model_name=model_name, api_key=api_key, **kwargs)
        cls._instances[cache_key] = instance
        
        return instance
    
    @classmethod
    def list_providers(cls) -> list:
        """List all registered embedding providers."""
        return list(cls._providers.keys())
    
    @classmethod
    def unregister(cls, name: str) -> bool:
        """Unregister an embedding provider."""
        name = name.lower()
        if name in cls._providers:
            del cls._providers[name]
            cls._instances = {k: v for k, v in cls._instances.items() if not k.startswith(f"{name}:")}
            print(f"Unregistered embedding provider: {name}")
            return True
        return False
    
    @classmethod
    def clear_cache(cls):
        """Clear all cached embedding provider instances."""
        cls._instances = {}

