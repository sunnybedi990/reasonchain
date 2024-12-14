import os
from reasonchain.llm_models.fine_tune import fine_tune_model
from reasonchain.utils.lazy_imports import os, openai, ollama, groq, dotenv, transformers

dotenv.load_dotenv()

class ModelManager:
    def __init__(self, api='openai',model_name='gpt-4', custom_model_path=None):
        """
        Initialize the ModelManager to handle various APIs.
        :param model_name: Default LLM model name.
        :param custom_model_path: Path to a custom fine-tuned model (optional).

        """
        self.api = api
        self.model_name = model_name
        self.custom_model_path = custom_model_path
        self.custom_model = None
        self.tokenizer = None

        if api == 'custom' and custom_model_path:
            self._load_custom_model(custom_model_path)      
        
    def _load_custom_model(self, custom_model_path):
            """
            Load a custom fine-tuned model and tokenizer.
            :param custom_model_path: Path to the custom model directory.
            """
            try:
                self.custom_model = transformers.AutoModelForCausalLM.from_pretrained(custom_model_path)
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(custom_model_path)
                print(f"Custom model loaded from {custom_model_path}")
            except Exception as e:
                print(f"Error loading custom model: {e}")

    @staticmethod
    def download_model(model_name, save_path="models"):
        """
        Download a model from Hugging Face and save it locally.
        :param model_name: Name of the model on Hugging Face (e.g., 'distilgpt2').
        :param save_path: Directory to save the model.
        """
        os.makedirs(save_path, exist_ok=True)
        model_path = os.path.join(save_path, model_name.replace("/", "_"))

        try:
            print(f"Downloading model: {model_name}")
            model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

            # Save the model and tokenizer
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)

            print(f"Model downloaded and saved to: {model_path}")
            return model_path
        except Exception as e:
            print(f"Error downloading model: {e}")
            return None
        
    def fine_tune(self, train_dataset, val_dataset, output_dir, num_train_epochs=3, per_device_train_batch_size=4):
        """
        Wrapper to fine-tune the custom model using the fine_tune module.
        """
        if not self.custom_model or not self.tokenizer:
            raise ValueError("Custom model or tokenizer not loaded.")
        return fine_tune_model(
            model=self.custom_model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size
        )

    def _generate_with_custom_model(self, prompt):
        """
        Generate a response using a custom fine-tuned model.
        :param prompt: Input text prompt.
        :return: Generated response.
        """
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            outputs = self.custom_model.generate(**inputs, max_length=200)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Error generating response with custom model: {e}")
            return "// No response from custom model."
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
            elif api == 'custom' and self.custom_model:
                return self._generate_with_custom_model(prompt)
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
        ollama_client = ollama.Client(host='http://localhost:11434')  # Ollama local client
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
        groq_client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))
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

