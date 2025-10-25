"""
Auto-registration of built-in LLM providers.

This module automatically registers all built-in providers when imported.
"""

from reasonchain.llm_models.provider_registry import LLMProviderRegistry
from reasonchain.llm_models.providers import (
    OpenAIProvider,
    GroqProvider,
    OllamaProvider,
    CustomModelProvider,
    AnthropicProvider
)


def register_builtin_providers():
    """Register all built-in LLM providers."""
    try:
        LLMProviderRegistry.register('openai', OpenAIProvider)
        LLMProviderRegistry.register('groq', GroqProvider)
        LLMProviderRegistry.register('ollama', OllamaProvider)
        LLMProviderRegistry.register('custom', CustomModelProvider)
        LLMProviderRegistry.register('anthropic', AnthropicProvider)
        print("Built-in LLM providers registered successfully")
    except Exception as e:
        print(f"Error registering built-in providers: {e}")


# Auto-register on import
register_builtin_providers()

