"""Groq LLM Provider Implementation"""

import os
from typing import List, Dict, Optional
from reasonchain.llm_models.base_provider import BaseLLMProvider
from reasonchain.utils.lazy_imports import groq, dotenv

dotenv.load_dotenv()


class GroqProvider(BaseLLMProvider):
    """
    Groq LLM provider implementation.
    
    Supports Groq's fast inference models including Llama, Mixtral, etc.
    """
    
    def __init__(self, model_name: str = "llama-3.1-8b-instant", api_key: Optional[str] = None, **kwargs):
        """
        Initialize Groq provider.
        
        Args:
            model_name (str): Groq model name
            api_key (str, optional): Groq API key
            **kwargs: Additional configuration
        """
        super().__init__(model_name, api_key, **kwargs)
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        
        if not self.api_key:
            raise ValueError("Groq API key is required. Set GROQ_API_KEY environment variable or pass api_key parameter.")
        
        try:
            self.client = groq.Groq(api_key=self.api_key)
        except Exception as e:
            print(f"Error initializing Groq client: {e}")
            raise
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate response from Groq model.
        
        Args:
            prompt (str): Input prompt
            **kwargs: Additional parameters
            
        Returns:
            str: Generated response
        """
        try:
            temperature = kwargs.get('temperature', self.config.get('temperature', 0.1))
            max_tokens = kwargs.get('max_tokens', self.config.get('max_tokens', 2000))
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Provide a concise and relevant answer to the user's query."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            if response.choices:
                return response.choices[0].message.content.strip()
            return "// No response from Groq."
        except Exception as e:
            print(f"[GroqProvider] Error generating response: {e}")
            return f"// Error: Unable to generate response with Groq: {e}"
    
    def generate_chat_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate chat response from Groq model.
        
        Args:
            messages (List[Dict]): List of message dicts
            **kwargs: Additional parameters
            
        Returns:
            str: Generated response
        """
        try:
            temperature = kwargs.get('temperature', self.config.get('temperature', 0.1))
            max_tokens = kwargs.get('max_tokens', self.config.get('max_tokens', 2000))
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            if response.choices:
                return response.choices[0].message.content.strip()
            return "// No response from Groq."
        except Exception as e:
            print(f"[GroqProvider] Error generating chat response: {e}")
            return f"// Error: Unable to generate chat response with Groq: {e}"
    
    def list_models(self) -> List[str]:
        """List available Groq models."""
        try:
            models = self.client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            print(f"Error listing Groq models: {e}")
            return [self.model_name]

