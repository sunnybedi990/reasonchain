import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class CoTPipeline:
    def __init__(self, agent):
        """
        Initialize the Chain of Thought pipeline.
        :param agent: The associated Agent instance.
        """
        self.agent = agent
        self.steps = []

    def add_step(self, description):
        """
        Add a reasoning step to the pipeline.
        :param description: Step description.
        """
        if not isinstance(description, str):
            description = str(description)
        self.steps.append(description)

    def execute(self, model_manager, combined_context=None):
        """
        Execute the reasoning steps.
        :param model_manager: Model manager for generating responses.
        :param combined_context: Context (short-term + RAG) for reasoning.
        :return: Result of reasoning.
        """
        print(f"[{self.agent.name}] Executing reasoning chain...")
        results = []
        for step in self.steps:
            # Add combined context to the step
            prompt = f"{step}\n\nContext: {combined_context}" if combined_context else step
            print(f"Step: {prompt}")
            # Use the LLM to generate output for each step
            response = model_manager.generate_response(prompt)
            results.append(response)
        return " -> ".join(results)
    
# Tree of Thought (ToT) Reasoning

class TreeOfThoughtPipeline:
    def __init__(self, agent):
        """
        Initialize the Tree of Thought pipeline.
        :param agent: The associated Agent instance.
        """
        self.agent = agent
        self.steps = []
        self.branching_factor = 2  # Number of branches at each step
        self.max_depth = 3  # Maximum depth of the tree

    def add_step(self, description):
        """
        Add a reasoning step to the pipeline.
        :param description: Step description.
        """
        self.steps.append(description)

    def evaluate_paths(self, paths):
        """
        Evaluate the generated paths to select the best ones.
        :param paths: List of (path, score) tuples.
        :return: Top-ranked paths based on scores.
        """
        return sorted(paths, key=lambda x: x[1], reverse=True)[:self.branching_factor]

    def execute(self, model_manager, combined_context=None):
        """
        Execute the reasoning steps using Tree of Thought.
        :param model_manager: Model manager for generating responses.
        :param combined_context: Context (short-term + RAG) for reasoning.
        :return: Final result after reasoning.
        """
        paths = [("", 0)]  # Initial path and score
        for depth in range(self.max_depth):
            new_paths = []
            for path, score in paths:
                for branch in range(self.branching_factor):
                    # Add combined context to the step
                    prompt = f"{path} Step {depth + 1}: {self.steps[min(depth, len(self.steps) - 1)]}"
                    if combined_context:
                        prompt += f"\n\nContext: {combined_context}"
                    response = model_manager.generate_response(prompt)
                    branch_score = self.score_response(response)
                    new_paths.append((f"{path} -> {response}", score + branch_score))
            paths = self.evaluate_paths(new_paths)
        return paths[0][0]  # Return the best path

    def score_response(self, response):
        """
        Score a response based on predefined criteria.
        :param response: Generated response.
        :return: Numeric score for the response.
        """
        # Placeholder: Define scoring logic (e.g., relevance, coherence)
        return len(response.split())  # Example: Length of the response as a proxy


# Parallel Reasoning Chains

import concurrent.futures

class ParallelCoTPipeline:
    def __init__(self, agent):
        """
        Initialize the Parallel Chain of Thought pipeline.
        :param agent: The associated Agent instance.
        """
        self.agent = agent
        self.steps = []

    def add_step(self, description):
        """
        Add a reasoning step to the pipeline.
        :param description: Step description.
        """
        self.steps.append(description)
        
    def execute(self, model_manager, combined_context=None):
        """
        Execute reasoning steps in parallel.
        :param model_manager: Model manager for generating responses.
        :param combined_context: Context (short-term + RAG) for reasoning.
        :return: Combined results from all parallel steps.
        """
        responses = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self.process_step, model_manager, step, combined_context): step
                for step in self.steps
            }
            for future in concurrent.futures.as_completed(futures):
                step = futures[future]
                try:
                    responses[step] = future.result()
                except Exception as e:
                    responses[step] = f"Error: {e}"

        # Aggregate the results into a cohesive format
        final_output = "\n".join([f"{step}: {result}" for step, result in responses.items()])
        return final_output

    def process_step(self, model_manager, step, combined_context=None):
        prompt = step
        if combined_context:
            prompt += f"\n\nContext: {combined_context}"
        return model_manager.generate_response(prompt)

    # def execute(self, model_manager):
    #     """
    #     Execute reasoning steps in parallel.
    #     :param model_manager: Model manager for generating responses.
    #     :return: Combined results from all parallel steps.
    #     """
    #     responses = {}
    #     with concurrent.futures.ThreadPoolExecutor() as executor:
    #         futures = {
    #             executor.submit(self.process_step, model_manager, step): step
    #             for step in self.steps
    #         }
    #         for future in concurrent.futures.as_completed(futures):
    #             step = futures[future]
    #             try:
    #                 responses[step] = future.result()
    #             except Exception as e:
    #                 responses[step] = f"Error: {e}"

    #     # Aggregate the results into a cohesive format
    #     final_output = "\n".join([f"{step}: {result}" for step, result in responses.items()])
    #     return final_output

    # def process_step(self, model_manager, step):
    #     if step == "Fetch data from the knowledge base.":
    #         # Integrate with RAG model
    #         query = "Fetch relevant knowledge for the current query"
    #         context = self.retrieve_from_rag(query)
    #         if context:
    #             return f"Fetched data: {context}"
    #         else:
    #             return "No relevant knowledge found in long-term memory."
    #     else:
    #         return model_manager.generate_response(step)

    def retrieve_from_rag(self, query):
        """
        Retrieve context from a RAG model or long-term memory.
        :param query: Query string.
        :return: Retrieved context.
        """
        # Placeholder for RAG integration
        return f"Simulated knowledge retrieval for query: {query}"


# Hybrid Pipeline combines the strengths of Chain of Thought (CoT), Tree of Thought (ToT), and Parallel Reasoning Chains.

class HybridCoTPipeline:
    def __init__(self, agent, complexity_evaluator=None):
        """
        Initialize the Hybrid Chain of Thought pipeline.
        :param agent: The associated Agent instance.
        """
        self.agent = agent
        self.cot_pipeline = CoTPipeline(agent)
        self.tot_pipeline = TreeOfThoughtPipeline(agent)
        self.parallel_pipeline = ParallelCoTPipeline(agent)
        self.steps = []
        self.complexity_evaluator = complexity_evaluator or self.default_complexity_evaluator


    @staticmethod
    def default_complexity_evaluator(step):
        """
        Dynamically evaluate the complexity of a step.
        :param step: Step description.
        :return: Complexity level ('low', 'medium', 'high').
        """
        if "fetch" in step.lower() or "retrieve" in step.lower():
            return "medium"
        elif "generate" in step.lower() or "evaluate" in step.lower():
            return "high"
        else:
            return "low"
            
    def add_step(self, description, complexity="low"):
        """
        Add a reasoning step with specified complexity.
        :param description: Step description.
        :param complexity: Complexity level ('low', 'medium', 'high').
        """
        complexity = self.complexity_evaluator(description)
        self.steps.append((description, complexity))
        
    def execute(self, model_manager, combined_context=None):
        """
        Execute the hybrid pipeline dynamically based on step complexity.
        :param model_manager: Model manager for generating responses.
        :param combined_context: Context (short-term + RAG) for reasoning.
        :return: Combined results from all steps.
        """
        final_results = []
        for step, complexity in self.steps:
            logging.info(f"Executing step: '{step}' with complexity: {complexity}")
            print(f"Executing step: '{step}' with complexity: {complexity}")

            if complexity == "low":
                # Use simple Chain of Thought
                self.cot_pipeline.add_step(step)
                result = self.cot_pipeline.execute(model_manager, combined_context)
            elif complexity == "medium":
                # Use Parallel Chain of Thought
                self.parallel_pipeline.add_step(step)
                result = self.parallel_pipeline.execute(model_manager, combined_context)
            elif complexity == "high":
                # Use Tree of Thought
                self.tot_pipeline.add_step(step)
                result = self.tot_pipeline.execute(model_manager, combined_context)
            else:
                raise ValueError(f"Unknown complexity level: {complexity}")
            
            final_results.append(result)
        return "\n".join(final_results)

