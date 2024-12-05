import os
import openai
from ollama import Client
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class ModelManager:
    def __init__(self, api='openai',model_name='gpt-4'):
        """
        Initialize the ModelManager to handle various APIs.
        :param model_name: Default LLM model name.
        :param api_key: API key for the default LLM service.
        """
        self.api = api
        self.model_name = model_name
       # self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        

    def generate_response(self, prompt, api=None, model_name=None):
        """
        Generate a response using the specified API.
        :param prompt: Text prompt for the LLM.
        :param api: API to use ('openai', 'ollama', or 'groq').
        :param selected_model: Specific model to use for the API.
        :return: Generated response or error message.
        """
        api = api or self.api
        selected_model = model_name or self.model_name
        try:
            if api == "openai":
                return self._generate_with_openai(prompt, selected_model)
            elif api == "ollama":
                return self._generate_with_ollama(prompt, selected_model)
            elif api == "groq":
                return self._generate_with_groq(prompt, selected_model)
            else:
                raise ValueError("Unsupported API specified.")
        except Exception as e:
            print(f"[ModelManager] Error generating response with {api}: {e}")
            return f"// Unable to generate response with {api}."

    def summarize(self, text, max_tokens=150, api=None, model_name=None):
        """
        Summarize a given text using the specified API.
        :param text: Text to summarize.
        :param max_tokens: Maximum tokens for the summary.
        :param api: API to use for summarization ('openai', 'ollama', or 'groq').
        :param selected_model: Specific model to use for the API.
        :return: Summarized text or error message.
        """
        prompt = f"Summarize the following:\n{text}"
        return self.generate_response(prompt, api=api, model_name=model_name)


    def _generate_with_openai(self, prompt, model):
        """
        Generate a response using OpenAI API.
        :param prompt: Text prompt.
        :param model: OpenAI model to use.
        :return: Generated response.
        """
        openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        try:
            response = openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[ModelManager] Error generating response with OpenAI: {e}")
            return "// No response from OpenAI."

    def _generate_with_ollama(self, prompt, model):
        """
        Generate a response using Ollama API.
        :param prompt: Text prompt.
        :param model: Ollama model to use.
        :return: Generated response.
        """
        ollama_client = Client(host='http://localhost:11434')  # Ollama local client
        try:
            response = ollama_client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": "Provide a concise and relevant answer to the user's query."},
                    {"role": "user", "content": prompt}
                ]
            )
            if response and 'message' in response:
                return response['message']['content']
        except Exception as e:
            print(f"[ModelManager] Error generating response with Ollama: {e}")
            return "// No response from Ollama."

    def _generate_with_groq(self, prompt, model):
        """
        Generate a response using Groq API.
        :param prompt: Text prompt.
        :param model: Groq model to use.
        :return: Generated response.
        """
        groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        try:
            response = groq_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Provide a concise and relevant answer to the user's query."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            if response.choices:
                return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[ModelManager] Error generating response with Groq: {e}")
            return "// No response from Groq."

    def list_available_models(self, api="openai"):
        """
        List available models for a specific API.
        :param api: API to query ('openai', 'ollama', or 'groq').
        :return: List of models or an error message.
        """
        try:
            if api == "openai":
                return self.openai_client.Model.list()["data"]
            elif api == "ollama":
                return self.ollama_client.list_models()
            elif api == "groq":
                return self.groq_client.models.list()
            else:
                return "// Error: Unsupported API specified."
        except Exception as e:
            print(f"[ModelManager] Error listing models for {api}: {e}")
            return f"// No models available for {api}."

