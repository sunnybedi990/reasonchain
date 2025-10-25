#!/usr/bin/env python3
"""
Custom Provider Example for ReasonChain

This example demonstrates how to:
1. Use built-in providers (OpenAI, Groq, Ollama, Anthropic)
2. Create and register custom LLM providers
3. Create and register custom embedding providers
4. Use the extensible provider architecture

Author: ReasonChain Team
"""

import os
import sys
from typing import List, Dict, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reasonchain.agent import Agent
from reasonchain.llm_models.provider_registry import LLMProviderRegistry, EmbeddingProviderRegistry
from reasonchain.llm_models.base_provider import BaseLLMProvider, BaseEmbeddingProvider


# ============================================================================
# EXAMPLE 1: Using Built-in Providers
# ============================================================================

def example_1_builtin_providers():
    """Example 1: Use built-in providers with the standard Agent interface."""
    print("\n" + "="*70)
    print("EXAMPLE 1: USING BUILT-IN PROVIDERS")
    print("="*70)
    
    # The familiar interface - now powered by the extensible provider system!
    agent_openai = Agent(name="OpenAI_Agent", model_name="gpt-4", api="openai")
    agent_groq = Agent(name="Groq_Agent", model_name="llama-3.1-8b-instant", api="groq")
    agent_ollama = Agent(name="Ollama_Agent", model_name="llama3.1:latest", api="ollama")
    
    # NEW: You can now use Anthropic too!
    # agent_anthropic = Agent(name="Claude_Agent", model_name="claude-3-opus-20240229", api="anthropic")
    
    print("\nAll agents created successfully using provider system!")
    print(f"Available providers: {LLMProviderRegistry.list_providers()}")


# ============================================================================
# EXAMPLE 2: Create a Custom LLM Provider (e.g., Cohere)
# ============================================================================

class CohereProvider(BaseLLMProvider):
    """
    Example custom provider for Cohere API.
    
    This shows how easy it is to add support for ANY LLM service!
    To use: pip install cohere
    """
    
    def __init__(self, model_name: str = "command", api_key: Optional[str] = None, **kwargs):
        super().__init__(model_name, api_key, **kwargs)
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        
        if not self.api_key:
            raise ValueError("Cohere API key required")
        
        try:
            import cohere
            self.client = cohere.Client(self.api_key)
        except ImportError:
            raise ImportError("Install cohere: pip install cohere")
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response using Cohere."""
        try:
            max_tokens = kwargs.get('max_tokens', 2000)
            temperature = kwargs.get('temperature', 0.7)
            
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response.generations[0].text
        except Exception as e:
            return f"// Error with Cohere: {e}"
    
    def generate_chat_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate chat response using Cohere."""
        # Convert messages to prompt
        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        return self.generate_response(prompt, **kwargs)


def example_2_custom_llm_provider():
    """Example 2: Register and use a custom LLM provider."""
    print("\n" + "="*70)
    print("EXAMPLE 2: CUSTOM LLM PROVIDER (COHERE)")
    print("="*70)
    
    # Register the custom provider
    LLMProviderRegistry.register('cohere', CohereProvider)
    print("✓ Registered Cohere provider")
    
    # Now you can use it just like built-in providers!
    # agent_cohere = Agent(name="Cohere_Agent", model_name="command", api="cohere")
    # response = agent_cohere.reason("What is machine learning?")
    
    print("\nCohere provider registered! You can now use it with:")
    print("  Agent(name='My_Agent', model_name='command', api='cohere')")


# ============================================================================
# EXAMPLE 3: Create a Custom Embedding Provider
# ============================================================================

class VoyageEmbeddingProvider(BaseEmbeddingProvider):
    """
    Example custom embedding provider for Voyage AI.
    
    Shows how to add support for ANY embedding service!
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
            self._dimension = 1024  # Voyage-01 dimension
        except ImportError:
            raise ImportError("Install voyageai: pip install voyageai")
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for single text."""
        try:
            result = self.client.embed([text], model=self.model_name)
            return result.embeddings[0]
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return []
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        try:
            result = self.client.embed(texts, model=self.model_name)
            return result.embeddings
        except Exception as e:
            print(f"Error generating batch embeddings: {e}")
            return []
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension


def example_3_custom_embedding_provider():
    """Example 3: Register and use a custom embedding provider."""
    print("\n" + "="*70)
    print("EXAMPLE 3: CUSTOM EMBEDDING PROVIDER (VOYAGE AI)")
    print("="*70)
    
    # Register the custom embedding provider
    EmbeddingProviderRegistry.register('voyage', VoyageEmbeddingProvider)
    print("✓ Registered Voyage AI embedding provider")
    
    # Now you can use it in your RAG pipeline!
    # from reasonchain.rag.vector.add_to_vector_db import add_data_to_vector_db
    # add_data_to_vector_db(
    #     file_paths=["document.pdf"],
    #     embedding_provider="voyage",
    #     embedding_model="voyage-01"
    # )
    
    print("\nVoyage AI provider registered! You can now use it with:")
    print("  add_data_to_vector_db(embedding_provider='voyage', embedding_model='voyage-01')")


# ============================================================================
# EXAMPLE 4: Local/Custom Model Provider
# ============================================================================

class LocalLlamaProvider(BaseLLMProvider):
    """
    Example provider for local Llama models.
    
    This could be a llama.cpp server, vLLM, or any local inference engine.
    """
    
    def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs):
        super().__init__(model_name, api_key, **kwargs)
        self.endpoint = kwargs.get('endpoint', 'http://localhost:8000')
        # Initialize your local model here
        print(f"Connected to local Llama model at {self.endpoint}")
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response from local model."""
        # Your local inference logic here
        return f"Response from local Llama model: {self.model_name}"
    
    def generate_chat_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate chat response from local model."""
        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        return self.generate_response(prompt, **kwargs)


def example_4_local_model_provider():
    """Example 4: Register and use a local model provider."""
    print("\n" + "="*70)
    print("EXAMPLE 4: LOCAL MODEL PROVIDER")
    print("="*70)
    
    # Register local model provider
    LLMProviderRegistry.register('local_llama', LocalLlamaProvider)
    print("✓ Registered local Llama provider")
    
    # Use it!
    # agent_local = Agent(
    #     name="Local_Agent",
    #     model_name="llama-7b",
    #     api="local_llama",
    #     endpoint="http://localhost:8000"
    # )
    
    print("\nLocal Llama provider registered! You can now use it with:")
    print("  Agent(name='Agent', model_name='llama-7b', api='local_llama', endpoint='http://localhost:8000')")


# ============================================================================
# EXAMPLE 5: List All Providers
# ============================================================================

def example_5_list_providers():
    """Example 5: List all available providers."""
    print("\n" + "="*70)
    print("EXAMPLE 5: LIST ALL AVAILABLE PROVIDERS")
    print("="*70)
    
    llm_providers = LLMProviderRegistry.list_providers()
    embedding_providers = EmbeddingProviderRegistry.list_providers()
    
    print("\n📋 Available LLM Providers:")
    for provider in llm_providers:
        print(f"  - {provider}")
    
    print("\n🔢 Available Embedding Providers:")
    for provider in embedding_providers:
        print(f"  - {provider}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("REASONCHAIN CUSTOM PROVIDER EXAMPLES")
    print("Demonstrating the Extensible Provider Architecture")
    print("="*70)
    
    try:
        # Example 1: Built-in providers
        example_1_builtin_providers()
        
        # Example 2: Custom LLM provider (Cohere)
        example_2_custom_llm_provider()
        
        # Example 3: Custom embedding provider (Voyage AI)
        example_3_custom_embedding_provider()
        
        # Example 4: Local model provider
        example_4_local_model_provider()
        
        # Example 5: List all providers
        example_5_list_providers()
        
        print("\n" + "="*70)
        print("✅ ALL EXAMPLES COMPLETED!")
        print("="*70)
        print("\nKey Takeaways:")
        print("1. ReasonChain now has a plugin architecture for LLMs and embeddings")
        print("2. Adding new providers is as simple as implementing a base class")
        print("3. No need to modify core code - just register your provider!")
        print("4. Full backward compatibility with existing code")
        print("5. Supports ANY LLM or embedding service (Anthropic, Cohere, Voyage, etc.)")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

