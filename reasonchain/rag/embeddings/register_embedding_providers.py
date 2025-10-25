"""
Auto-registration of built-in embedding providers.

This module automatically registers all built-in embedding providers when imported.
"""

from reasonchain.llm_models.provider_registry import EmbeddingProviderRegistry
from reasonchain.rag.embeddings.providers import (
    SentenceTransformersProvider,
    OpenAIEmbeddingsProvider,
    HuggingFaceProvider
)


def register_builtin_embedding_providers():
    """Register all built-in embedding providers."""
    try:
        EmbeddingProviderRegistry.register('sentence_transformers', SentenceTransformersProvider)
        EmbeddingProviderRegistry.register('openai', OpenAIEmbeddingsProvider)
        EmbeddingProviderRegistry.register('hugging_face', HuggingFaceProvider)
        EmbeddingProviderRegistry.register('huggingface', HuggingFaceProvider)  # Alias
        print("Built-in embedding providers registered successfully")
    except Exception as e:
        print(f"Error registering built-in embedding providers: {e}")


# Auto-register on import
register_builtin_embedding_providers()

