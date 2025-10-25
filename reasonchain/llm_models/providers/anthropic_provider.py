"""
Anthropic (Claude) LLM Provider Implementation

This is an example of how easy it is to add support for new LLM providers.
Users can follow this pattern to add support for any LLM service.
"""

import os
from typing import List, Dict, Optional
from reasonchain.llm_models.base_provider import BaseLLMProvider
from reasonchain.utils.lazy_imports import dotenv

dotenv.load_dotenv()


class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic (Claude) LLM provider implementation.
    
    Example provider showing how to add support for new LLM services.
    To use this, install: pip install anthropic
    """
    
    def __init__(self, model_name: str = "claude-3-opus-20240229", api_key: Optional[str] = None, **kwargs):
        """
        Initialize Anthropic provider.
        
        Args:
            model_name (str): Claude model name
            api_key (str, optional): Anthropic API key
            **kwargs: Additional configuration
        """
        super().__init__(model_name, api_key, **kwargs)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        
        if not self.api_key:
            raise ValueError("Anthropic API key is required. Set ANTHROPIC_API_KEY environment variable or pass api_key parameter.")
        
        try:
            # Import anthropic only when needed (lazy import)
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("Anthropic library not installed. Install with: pip install anthropic")
        except Exception as e:
            print(f"Error initializing Anthropic client: {e}")
            raise
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate response from Claude model.
        
        Args:
            prompt (str): Input prompt
            **kwargs: Additional parameters
            
        Returns:
            str: Generated response
        """
        try:
            max_tokens = kwargs.get('max_tokens', self.config.get('max_tokens', 2000))
            temperature = kwargs.get('temperature', self.config.get('temperature', 0.7))
            
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return message.content[0].text
        except Exception as e:
            print(f"[AnthropicProvider] Error generating response: {e}")
            return f"// Error: Unable to generate response with Anthropic: {e}"
    
    def generate_chat_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate chat response from Claude model.
        
        Args:
            messages (List[Dict]): List of message dicts
            **kwargs: Additional parameters
            
        Returns:
            str: Generated response
        """
        try:
            max_tokens = kwargs.get('max_tokens', self.config.get('max_tokens', 2000))
            temperature = kwargs.get('temperature', self.config.get('temperature', 0.7))
            
            # Filter out system messages (Claude handles them differently)
            user_messages = [msg for msg in messages if msg['role'] != 'system']
            system_content = next((msg['content'] for msg in messages if msg['role'] == 'system'), None)
            
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_content if system_content else "You are a helpful AI assistant.",
                messages=user_messages
            )
            
            return message.content[0].text
        except Exception as e:
            print(f"[AnthropicProvider] Error generating chat response: {e}")
            return f"// Error: Unable to generate chat response with Anthropic: {e}"
    
    def list_models(self) -> List[str]:
        """List available Claude models."""
        # Anthropic doesn't provide a models endpoint, so return common models
        return [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-2.1",
            "claude-2.0"
        ]

