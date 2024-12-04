# reasonchain/model_manager.py
import openai

class ModelManager:
    def __init__(self, model_name="gpt-4", api_key=None):
        """
        Initialize the model manager.
        :param model_name: Name of the LLM model (e.g., gpt-4).
        :param api_key: API key for the LLM service.
        """
        self.model_name = model_name
        self.api_key = api_key
        self.openai_client = openai.OpenAI(api_key=api_key)

        openai.api_key = self.api_key

    def generate_response(self, prompt):
        """
        Generate a response using the LLM.
        :param prompt: Text prompt for the LLM.
        :return: Generated response.
        """
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[ModelManager] Error generating response: {e}")
            return "Sorry, I couldn't process that."

    def summarize(self, text, max_tokens=150):
            """
            Summarize a given text using the LLM.
            :param text: Text to summarize.
            :param max_tokens: Maximum number of tokens for the summary.
            :return: Summarized text.
            """
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that summarizes text."},
                        {"role": "user", "content": f"Summarize the following:\n{text}"}
                    ],
                    temperature=0.7,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"[ModelManager] Error summarizing text: {e}")
                return "Sorry, I couldn't summarize that."