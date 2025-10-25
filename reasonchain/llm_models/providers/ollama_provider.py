"""Ollama LLM Provider Implementation"""

from typing import List, Dict, Optional
from reasonchain.llm_models.base_provider import BaseLLMProvider
from reasonchain.utils.lazy_imports import ollama


class OllamaProvider(BaseLLMProvider):
    """
    Ollama LLM provider implementation.
    
    Supports local Ollama models including Llama, Mistral, etc.
    """
    
    def __init__(self, model_name: str = "llama3.1:latest", api_key: Optional[str] = None, **kwargs):
        """
        Initialize Ollama provider.
        
        Args:
            model_name (str): Ollama model name
            api_key (str, optional): Not used for Ollama
            **kwargs: Additional configuration (host, port, etc.)
        """
        super().__init__(model_name, api_key, **kwargs)
        host = kwargs.get('host', 'http://localhost:11434')
        
        try:
            self.client = ollama.Client(host=host)
        except Exception as e:
            print(f"Error initializing Ollama client: {e}")
            raise
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate response from Ollama model.
        
        Args:
            prompt (str): Input prompt
            **kwargs: Additional parameters
            
        Returns:
            str: Generated response
        """
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Provide a concise and relevant answer to the user's query."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            if response and 'message' in response:
                return response['message']['content']
            return "// No response from Ollama."
        except Exception as e:
            print(f"[OllamaProvider] Error generating response: {e}")
            return f"// Error: Unable to generate response with Ollama: {e}"
    
    def generate_chat_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate chat response from Ollama model.
        
        Args:
            messages (List[Dict]): List of message dicts
            **kwargs: Additional parameters
            
        Returns:
            str: Generated response
        """
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=messages
            )
            
            if response and 'message' in response:
                return response['message']['content']
            return "// No response from Ollama."
        except Exception as e:
            print(f"[OllamaProvider] Error generating chat response: {e}")
            return f"// Error: Unable to generate chat response with Ollama: {e}"
    
    def list_models(self) -> List[str]:
        """List available Ollama models."""
        try:
            models = self.client.list()
            if models and 'models' in models:
                return [model['name'] for model in models['models']]
            return [self.model_name]
        except Exception as e:
            print(f"Error listing Ollama models: {e}")
            return [self.model_name]

