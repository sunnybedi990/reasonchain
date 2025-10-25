"""OpenAI LLM Provider Implementation"""

import os
from typing import List, Dict, Optional
from reasonchain.llm_models.base_provider import BaseLLMProvider
from reasonchain.utils.lazy_imports import openai, dotenv

dotenv.load_dotenv()


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI LLM provider implementation.
    
    Supports all OpenAI chat models including GPT-4, GPT-3.5, etc.
    """
    
    def __init__(self, model_name: str = "gpt-4", api_key: Optional[str] = None, **kwargs):
        """
        Initialize OpenAI provider.
        
        Args:
            model_name (str): OpenAI model name (default: gpt-4)
            api_key (str, optional): OpenAI API key (uses env var if not provided)
            **kwargs: Additional configuration
        """
        super().__init__(model_name, api_key, **kwargs)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        try:
            self.client = openai.OpenAI(api_key=self.api_key)
        except Exception as e:
            print(f"Error initializing OpenAI client: {e}")
            raise
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate response from OpenAI model.
        
        Args:
            prompt (str): Input prompt
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            str: Generated response
        """
        try:
            temperature = kwargs.get('temperature', self.config.get('temperature', 0.7))
            max_tokens = kwargs.get('max_tokens', self.config.get('max_tokens', 2000))
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[OpenAIProvider] Error generating response: {e}")
            return f"// Error: Unable to generate response with OpenAI: {e}"
    
    def generate_chat_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate chat response from OpenAI model.
        
        Args:
            messages (List[Dict]): List of message dicts with 'role' and 'content'
            **kwargs: Additional parameters
            
        Returns:
            str: Generated response
        """
        try:
            temperature = kwargs.get('temperature', self.config.get('temperature', 0.7))
            max_tokens = kwargs.get('max_tokens', self.config.get('max_tokens', 2000))
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[OpenAIProvider] Error generating chat response: {e}")
            return f"// Error: Unable to generate chat response with OpenAI: {e}"
    
    def list_models(self) -> List[str]:
        """List available OpenAI models."""
        try:
            models = self.client.models.list()
            return [model.id for model in models.data if 'gpt' in model.id.lower()]
        except Exception as e:
            print(f"Error listing OpenAI models: {e}")
            return [self.model_name]

