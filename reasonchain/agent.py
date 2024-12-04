from .memory import Memory
from .model_manager import ModelManager

class Agent:
    def __init__(self, name, model_name="gpt-4", api_key=None, memory=None):
        """
        Initialize the agent.
        :param name: Name of the agent.
        :param model: LLM model instance.
        :param memory: Memory instance for storing context.
        """
        self.name = name
        self.model_manager = ModelManager(model_name=model_name, api_key=api_key)
        self.memory = memory or Memory()  # Use Memory class

    def observe(self, input_data):
        """
        Process input data.
        :param input_data: Raw input for the agent.
        :return: Processed data.
        """
        self.memory.store_short_term(input_data)  # Store in short-term memory
        print(f"[{self.name}] Observing input: {input_data}")
        return input_data

    def reason(self, cot_pipeline):
        """
        Execute the CoT pipeline for reasoning.
        :param cot_pipeline: CoTPipeline instance.
        :return: Reasoned response.
        """
        print(f"[{self.name}] Starting reasoning process...")
        context = self.memory.retrieve_short_term()
        long_term_context = self.memory.retrieve_long_term(" ".join(context))
        print(f"[{self.name}] Context from short-term memory: {context}")
        print(f"[{self.name}] Retrieved from long-term memory: {long_term_context}")
        return cot_pipeline.execute(self.model_manager)

    def act(self, response):
        """
        Act based on the response.
        :param response: Reasoned output.
        """
        self.memory.store_long_term(response)  # Store in long-term memory
        print(f"[{self.name}] Acting on response: {response}")
        return response