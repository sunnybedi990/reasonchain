import os

# Default embedding configurations for popular models
DEFAULT_EMBEDDING_CONFIGS = {
    "sentence_transformers": {
        "all-mpnet-base-v2": {"dimension": 768, "description": "High-quality sentence embeddings for semantic search and clustering."},
        "all-MiniLM-L6-v2": {"dimension": 384, "description": "Efficient model balancing performance and computational needs."},
        "paraphrase-MiniLM-L12-v2": {"dimension": 384, "description": "Captures paraphrastic relationships for paraphrase detection."},
        
    },
    "openai": {
        "text-embedding-ada-002": {"dimension": 1536, "description": "High-quality embeddings for various NLP tasks via OpenAI API."},
        "text-embedding-3-small": {"dimension": 1536, "description": "OpenAI's latest small embedding model."},
        "text-embedding-3-large": {"dimension": 3072, "description": "OpenAI's latest large embedding model."}
    },
    "hugging_face": {
        "distilbert-base-uncased": {"dimension": 768, "description": "Distilled BERT with faster inference and reduced model size."},
        "roberta-base": {"dimension": 768, "description": "Optimized BERT model, especially for nuanced language understanding."}
    },
    "google_use": {
        "universal-sentence-encoder": {"dimension": 512, "description": "Google's universal sentence encoder for transfer learning tasks."}
    },
    "elmo": {
        "elmo-original": {"dimension": 1024, "description": "Deep contextualized word representations from ELMo for NLP tasks."}
    },
    "glove": {
        "glove.6B.300d": {"dimension": 300, "description": "Word-level embeddings capturing global statistical information."}
    },
    "fasttext": {
        "fasttext-wiki-news-subwords-300": {"dimension": 300, "description": "Word embeddings with subword information, beneficial for rare words."}
    },
    "bert": {
        "bert-base-uncased": {"dimension": 768, "description": "Bidirectional BERT for general NLP tasks with contextual understanding."},
        "bert-large-uncased": {"dimension": 1024, "description": "Larger BERT model offering deeper contextual representations."}
    },
    "albert": {
        "albert-base-v2": {"dimension": 768, "description": "Lite version of BERT, efficient for resource-constrained tasks."}
    },
    "xlnet": {
        "xlnet-base-cased": {"dimension": 768, "description": "Generalized autoregressive model excelling in capturing bidirectional context."}
    },
    "gpt2": {
        "gpt2-medium": {"dimension": 1024, "description": "Generative model from GPT-2, good for text generation tasks."}
    },
    "t5": {
        "t5-base": {"dimension": 768, "description": "Text-to-Text Transfer Transformer for diverse NLP tasks like translation and summarization."}
    },
    # Custom provider for user-defined models
    "custom": {},
    "local": {},
    "fine_tuned": {}
}

# Global registry for custom models
custom_embedding_configs = DEFAULT_EMBEDDING_CONFIGS.copy()

def register_custom_model(provider, model_name, dimension, description="Custom embedding model", model_path=None, **kwargs):
    """
    Register a custom embedding model configuration.
    
    Args:
        provider (str): Provider name (e.g., 'custom', 'local', 'fine_tuned', 'hugging_face')
        model_name (str): Model identifier or name
        dimension (int): Embedding dimension
        description (str): Model description
        model_path (str, optional): Local path to the model (for local models)
        **kwargs: Additional configuration parameters
    
    Returns:
        dict: The registered model configuration
    """
    if provider not in custom_embedding_configs:
        custom_embedding_configs[provider] = {}
    
    config = {
        "dimension": dimension,
        "description": description,
        **kwargs
    }
    
    if model_path:
        config["model_path"] = model_path
    
    custom_embedding_configs[provider][model_name] = config
    print(f"Registered custom model: {provider}/{model_name} with dimension {dimension}")
    return config

def register_huggingface_model(model_name, dimension, description="HuggingFace embedding model", **kwargs):
    """
    Register a HuggingFace model (including fine-tuned ones).
    
    Args:
        model_name (str): HuggingFace model identifier (e.g., 'user/my-fine-tuned-model')
        dimension (int): Embedding dimension
        description (str): Model description
        **kwargs: Additional configuration parameters
    """
    return register_custom_model("hugging_face", model_name, dimension, description, **kwargs)

def register_local_model(model_name, model_path, dimension, description="Local embedding model", **kwargs):
    """
    Register a local embedding model.
    
    Args:
        model_name (str): Local model identifier
        model_path (str): Path to the local model directory
        dimension (int): Embedding dimension
        description (str): Model description
        **kwargs: Additional configuration parameters
    """
    if not os.path.exists(model_path):
        raise ValueError(f"Model path does not exist: {model_path}")
    
    return register_custom_model("local", model_name, dimension, description, model_path=model_path, **kwargs)

def register_fine_tuned_model(model_name, base_model, dimension, description="Fine-tuned embedding model", model_path=None, **kwargs):
    """
    Register a fine-tuned embedding model.
    
    Args:
        model_name (str): Fine-tuned model identifier
        base_model (str): Base model that was fine-tuned
        dimension (int): Embedding dimension
        description (str): Model description
        model_path (str, optional): Path to fine-tuned model (local or HF repo)
        **kwargs: Additional configuration parameters
    """
    config = {
        "base_model": base_model,
        **kwargs
    }
    
    if model_path:
        config["model_path"] = model_path
    
    return register_custom_model("fine_tuned", model_name, dimension, description, **config)

def get_embedding_config(provider, model_name):
    """
    Retrieve the configuration for a given provider and model name.
    
    Args:
        provider (str): Embedding provider
        model_name (str): Model name
        
    Returns:
        dict: Model configuration
        
    Raises:
        ValueError: If model is not found in configuration
    """
    config = custom_embedding_configs.get(provider, {}).get(model_name)
    if not config:
        raise ValueError(f"Model {model_name} for provider {provider} not found in configuration.")
    return config

def list_available_models(provider=None):
    """
    List all available models, optionally filtered by provider.
    
    Args:
        provider (str, optional): Filter by specific provider
        
    Returns:
        dict: Available models organized by provider
    """
    if provider:
        if provider not in custom_embedding_configs:
            return {}
        return {provider: custom_embedding_configs[provider]}
    
    return custom_embedding_configs

def get_model_dimension(provider, model_name):
    """
    Get the embedding dimension for a specific model.
    
    Args:
        provider (str): Embedding provider
        model_name (str): Model name
        
    Returns:
        int: Embedding dimension
    """
    config = get_embedding_config(provider, model_name)
    return config["dimension"]

def remove_custom_model(provider, model_name):
    """
    Remove a custom model from the registry.
    
    Args:
        provider (str): Provider name
        model_name (str): Model name
        
    Returns:
        bool: True if model was removed, False if not found
    """
    if provider in custom_embedding_configs and model_name in custom_embedding_configs[provider]:
        del custom_embedding_configs[provider][model_name]
        print(f"Removed custom model: {provider}/{model_name}")
        return True
    return False

# Example usage and registration of common custom models
if __name__ == "__main__":
    # Example: Register a fine-tuned model from HuggingFace
    register_huggingface_model(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        384,
        "Multilingual paraphrase model"
    )
    
    # Example: Register a local fine-tuned model
    try:
        register_local_model(
            "my-domain-model",
            "/path/to/my/fine-tuned-model",
            768,
            "Domain-specific fine-tuned model"
        )
    except ValueError as e:
        print(f"Could not register local model: {e}")
    
    # Example: Register a fine-tuned model
    register_fine_tuned_model(
        "my-finance-model",
        "sentence-transformers/all-mpnet-base-v2",
        768,
        "Finance domain fine-tuned model",
        model_path="user/finance-embedding-model"
    )
    
    # List all available models
    print("\nAvailable models:")
    for provider, models in list_available_models().items():
        if models:  # Only show providers with models
            print(f"{provider}:")
            for model_name, config in models.items():
                print(f"  - {model_name}: {config['dimension']}D - {config['description']}")

# Backward compatibility
embedding_configs = custom_embedding_configs
