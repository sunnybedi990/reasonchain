# ReasonChain Extensibility Guide

## 🎯 Overview

ReasonChain now features a **powerful plugin architecture** that allows you to add support for ANY LLM or embedding service without modifying core code. This document explains how to use and extend the system.

## 🔌 Plugin Architecture

### What Problem Does This Solve?

**Before**: ReasonChain was limited to 3-4 hardcoded LLM services (OpenAI, Groq, Ollama). Adding new services required modifying core code.

**After**: ReasonChain now has an open, extensible architecture. Adding support for a new LLM service is as simple as implementing a base class and registering it.

### Core Components

```
reasonchain/
├── llm_models/
│   ├── base_provider.py         # Abstract interfaces
│   ├── provider_registry.py     # Provider registration system
│   ├── register_providers.py    # Auto-registration
│   └── providers/               # Built-in implementations
│       ├── openai_provider.py
│       ├── groq_provider.py
│       ├── ollama_provider.py
│       ├── custom_provider.py
│       └── anthropic_provider.py
```

## 🚀 Quick Start

### Using Built-in Providers

No changes needed! Your existing code works as-is:

```python
from reasonchain.agent import Agent

# All of these work out of the box
agent1 = Agent(name="GPT", model_name="gpt-4", api="openai")
agent2 = Agent(name="Groq", model_name="llama-3.1-8b-instant", api="groq")
agent3 = Agent(name="Ollama", model_name="llama3.1:latest", api="ollama")
agent4 = Agent(name="Claude", model_name="claude-3-opus-20240229", api="anthropic")
```

### Adding a Custom LLM Provider

Add support for ANY LLM service in 3 simple steps:

```python
from reasonchain.llm_models.base_provider import BaseLLMProvider
from reasonchain.llm_models.provider_registry import LLMProviderRegistry

# Step 1: Implement the provider
class CohereProvider(BaseLLMProvider):
    def __init__(self, model_name, api_key=None, **kwargs):
        super().__init__(model_name, api_key, **kwargs)
        import cohere
        self.client = cohere.Client(api_key=api_key or os.getenv("COHERE_API_KEY"))
    
    def generate_response(self, prompt, **kwargs):
        response = self.client.generate(
            model=self.model_name,
            prompt=prompt,
            max_tokens=kwargs.get('max_tokens', 2000)
        )
        return response.generations[0].text
    
    def generate_chat_response(self, messages, **kwargs):
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        return self.generate_response(prompt, **kwargs)

# Step 2: Register it
LLMProviderRegistry.register('cohere', CohereProvider)

# Step 3: Use it!
agent = Agent(name="Cohere_Agent", model_name="command", api="cohere")
response = agent.reason("What is machine learning?")
```

### Adding a Custom Embedding Provider

Same simple pattern for embeddings:

```python
from reasonchain.llm_models.base_provider import BaseEmbeddingProvider
from reasonchain.llm_models.provider_registry import EmbeddingProviderRegistry

class VoyageEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, model_name, api_key=None, **kwargs):
        super().__init__(model_name, api_key, **kwargs)
        import voyageai
        self.client = voyageai.Client(api_key=api_key)
        self._dimension = 1024
    
    def embed_text(self, text):
        result = self.client.embed([text], model=self.model_name)
        return result.embeddings[0]
    
    def embed_batch(self, texts, batch_size=32):
        result = self.client.embed(texts, model=self.model_name)
        return result.embeddings
    
    def get_dimension(self):
        return self._dimension

# Register and use
EmbeddingProviderRegistry.register('voyage', VoyageEmbeddingProvider)
```

## 📚 Built-in Providers

### LLM Providers

| Provider | Models | Description |
|----------|--------|-------------|
| **openai** | GPT-4, GPT-3.5-turbo, etc. | OpenAI's chat models |
| **groq** | Llama 3.1, Mixtral, etc. | Fast inference with Groq |
| **ollama** | Llama, Mistral, etc. | Local models via Ollama |
| **anthropic** | Claude 3 Opus, Sonnet, Haiku | Anthropic's Claude models |
| **custom** | Any local model | Load custom/fine-tuned HuggingFace models |

### Embedding Providers

| Provider | Models | Description |
|----------|--------|-------------|
| **sentence_transformers** | all-mpnet-base-v2, all-MiniLM-L6-v2, etc. | All sentence-transformers models |
| **openai** | text-embedding-ada-002, text-embedding-3-* | OpenAI's embedding models |
| **hugging_face** | BERT, RoBERTa, DistilBERT, custom | Any HuggingFace transformer model |

## 🎨 Provider Interface

### BaseLLMProvider

Required methods:

```python
class BaseLLMProvider(ABC):
    @abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response from a simple prompt."""
        pass
    
    @abstractmethod
    def generate_chat_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response from chat messages."""
        pass
```

Optional methods:

```python
    def summarize(self, text: str, max_tokens: int = 150, **kwargs) -> str:
        """Summarize text (has default implementation)."""
        pass
    
    def list_models(self) -> List[str]:
        """List available models."""
        pass
    
    def validate_config(self) -> bool:
        """Validate configuration."""
        pass
```

### BaseEmbeddingProvider

Required methods:

```python
class BaseEmbeddingProvider(ABC):
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for single text."""
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        pass
```

## 🔧 Advanced Usage

### Provider Configuration

Pass custom configuration to providers:

```python
agent = Agent(
    name="Custom_Agent",
    model_name="gpt-4",
    api="openai",
    temperature=0.9,
    max_tokens=4000,
    custom_param="value"
)
```

### Using Pre-initialized Providers

```python
from reasonchain.llm_models.provider_registry import LLMProviderRegistry

# Get provider directly
provider = LLMProviderRegistry.get_provider(
    name="openai",
    model_name="gpt-4",
    api_key="your-key"
)

# Use with ModelManager
from reasonchain.llm_models.model_manager import ModelManager

model_manager = ModelManager(provider=provider)
response = model_manager.generate_response("Hello!")
```

### List Available Providers

```python
from reasonchain.llm_models.provider_registry import LLMProviderRegistry, EmbeddingProviderRegistry

llm_providers = LLMProviderRegistry.list_providers()
embedding_providers = EmbeddingProviderRegistry.list_providers()

print(f"LLM Providers: {llm_providers}")
print(f"Embedding Providers: {embedding_providers}")
```

## 🌟 Real-World Examples

### Example 1: AI21 Labs Provider

```python
class AI21Provider(BaseLLMProvider):
    def __init__(self, model_name="j2-ultra", api_key=None, **kwargs):
        super().__init__(model_name, api_key, **kwargs)
        import ai21
        ai21.api_key = api_key or os.getenv("AI21_API_KEY")
        self.ai21 = ai21
    
    def generate_response(self, prompt, **kwargs):
        response = self.ai21.Completion.execute(
            model=self.model_name,
            prompt=prompt,
            maxTokens=kwargs.get('max_tokens', 2000)
        )
        return response['completions'][0]['data']['text']
    
    def generate_chat_response(self, messages, **kwargs):
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        return self.generate_response(prompt, **kwargs)

LLMProviderRegistry.register('ai21', AI21Provider)
```

### Example 2: Local vLLM Server

```python
class VLLMProvider(BaseLLMProvider):
    def __init__(self, model_name, api_key=None, **kwargs):
        super().__init__(model_name, api_key, **kwargs)
        self.endpoint = kwargs.get('endpoint', 'http://localhost:8000')
        import requests
        self.requests = requests
    
    def generate_response(self, prompt, **kwargs):
        response = self.requests.post(
            f"{self.endpoint}/generate",
            json={
                "prompt": prompt,
                "model": self.model_name,
                "max_tokens": kwargs.get('max_tokens', 2000)
            }
        )
        return response.json()['text']
    
    def generate_chat_response(self, messages, **kwargs):
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        return self.generate_response(prompt, **kwargs)

LLMProviderRegistry.register('vllm', VLLMProvider)

# Use it
agent = Agent(
    name="Local_LLM",
    model_name="llama-7b",
    api="vllm",
    endpoint="http://localhost:8000"
)
```

### Example 3: Cohere Embedding Provider

```python
class CohereEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, model_name="embed-english-v3.0", api_key=None, **kwargs):
        super().__init__(model_name, api_key, **kwargs)
        import cohere
        self.client = cohere.Client(api_key or os.getenv("COHERE_API_KEY"))
        self._dimension = 1024
    
    def embed_text(self, text):
        response = self.client.embed(
            texts=[text],
            model=self.model_name,
            input_type="search_document"
        )
        return response.embeddings[0]
    
    def embed_batch(self, texts, batch_size=96):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            response = self.client.embed(
                texts=batch,
                model=self.model_name,
                input_type="search_document"
            )
            all_embeddings.extend(response.embeddings)
        return all_embeddings
    
    def get_dimension(self):
        return self._dimension

EmbeddingProviderRegistry.register('cohere', CohereEmbeddingProvider)

# Use it!
add_data_to_vector_db(
    file_paths=["documents.pdf"],
    embedding_provider="cohere",
    embedding_model="embed-english-v3.0"
)
```

### Example 4: Custom Embedding Service

```python
class CustomEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, model_name, api_key=None, **kwargs):
        super().__init__(model_name, api_key, **kwargs)
        self.endpoint = kwargs.get('endpoint', 'http://localhost:5000')
        self._dimension = kwargs.get('dimension', 768)
    
    def embed_text(self, text):
        import requests
        response = requests.post(
            f"{self.endpoint}/embed",
            json={"text": text, "model": self.model_name}
        )
        return response.json()['embedding']
    
    def embed_batch(self, texts, batch_size=32):
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = [self.embed_text(text) for text in batch]
            embeddings.extend(batch_embeddings)
        return embeddings
    
    def get_dimension(self):
        return self._dimension

EmbeddingProviderRegistry.register('custom_embed', CustomEmbeddingProvider)
```

## 🎁 Benefits

1. **No Vendor Lock-in**: Switch between providers easily
2. **Future-Proof**: Support for new services added in minutes
3. **Plugin Ecosystem**: Share custom providers as packages
4. **Backward Compatible**: All existing code works unchanged
5. **Type Safe**: Abstract base classes ensure consistency
6. **Testable**: Easy to mock providers for testing
7. **Flexible**: Pass custom configuration to providers
8. **Cached**: Provider instances are cached for performance

## 📦 Distributing Custom Providers

Create a package for your custom providers:

```python
# my_reasonchain_providers/cohere_provider.py
from reasonchain.llm_models.base_provider import BaseLLMProvider

class CohereProvider(BaseLLMProvider):
    # ... implementation

# my_reasonchain_providers/__init__.py
from reasonchain.llm_models.provider_registry import LLMProviderRegistry
from .cohere_provider import CohereProvider

def register_providers():
    LLMProviderRegistry.register('cohere', CohereProvider)

# Auto-register on import
register_providers()
```

Users can then:

```python
# Install your package
# pip install my-reasonchain-providers

# Use it
import my_reasonchain_providers  # Auto-registers
from reasonchain.agent import Agent

agent = Agent(name="Cohere_Agent", model_name="command", api="cohere")
```

## 🤝 Contributing Providers

Want to contribute a provider to ReasonChain? 

1. Implement the provider following the interface
2. Add it to `reasonchain/llm_models/providers/`
3. Register it in `register_providers.py`
4. Add tests and documentation
5. Submit a pull request!

## 📖 Full Example

See `examples/custom_provider_example.py` for a comprehensive demonstration of:
- Using built-in providers
- Creating custom LLM providers
- Creating custom embedding providers
- Local model providers
- Listing providers

## 🆘 Troubleshooting

### Provider Not Found

```python
# Error: Provider 'cohere' not found
# Solution: Register the provider first
LLMProviderRegistry.register('cohere', CohereProvider)
```

### Import Errors

```python
# Error: Module 'cohere' not found
# Solution: Install required dependencies
# pip install cohere
```

### API Key Issues

```python
# Error: API key required
# Solution: Set environment variable or pass explicitly
agent = Agent(name="Agent", model_name="command", api="cohere", api_key="your-key")
```

## 🎯 Conclusion

The extensible provider architecture makes ReasonChain truly open and future-proof. You can now:

- ✅ Use ANY LLM service (Anthropic, Cohere, AI21, local models, etc.)
- ✅ Add support for new services without modifying core code
- ✅ Share custom providers as packages
- ✅ Maintain full backward compatibility
- ✅ Switch providers easily with no code changes

**ReasonChain is no longer limited to specific LLM services - it's a universal reasoning framework that works with any AI model!**

