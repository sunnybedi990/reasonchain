"""Custom Model Provider Implementation"""

from typing import List, Dict, Optional
from reasonchain.llm_models.base_provider import BaseLLMProvider
from reasonchain.utils.lazy_imports import transformers


class CustomModelProvider(BaseLLMProvider):
    """
    Custom/Local model provider implementation.
    
    Supports loading custom fine-tuned models or local Hugging Face models.
    """
    
    def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs):
        """
        Initialize custom model provider.
        
        Args:
            model_name (str): Model name or path
            api_key (str, optional): Not used for custom models
            **kwargs: Additional configuration (model_path, etc.)
        """
        super().__init__(model_name, api_key, **kwargs)
        self.model_path = kwargs.get('model_path', model_name)
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the custom model and tokenizer."""
        try:
            print(f"Loading custom model from {self.model_path}...")
            self.model = transformers.AutoModelForCausalLM.from_pretrained(self.model_path)
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_path)
            print(f"Custom model loaded successfully: {self.model_name}")
        except Exception as e:
            print(f"Error loading custom model: {e}")
            raise
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate response from custom model.
        
        Args:
            prompt (str): Input prompt
            **kwargs: Additional parameters
            
        Returns:
            str: Generated response
        """
        try:
            max_length = kwargs.get('max_length', 200)
            
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            outputs = self.model.generate(**inputs, max_length=max_length)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return response
        except Exception as e:
            print(f"[CustomModelProvider] Error generating response: {e}")
            return f"// Error: Unable to generate response with custom model: {e}"
    
    def generate_chat_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate chat response from custom model.
        
        Args:
            messages (List[Dict]): List of message dicts
            **kwargs: Additional parameters
            
        Returns:
            str: Generated response
        """
        # Convert messages to a single prompt
        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        return self.generate_response(prompt, **kwargs)
    
    def list_models(self) -> List[str]:
        """List available custom models."""
        return [self.model_name]

