from reasonchain import Agent
from reasonchain.cot_pipeline import ParallelCoTPipeline

agent = Agent(name="ParallelBot", model_name="gpt-4o", api='openai')

# Initialize Parallel CoT pipeline
pipeline = ParallelCoTPipeline(agent=agent)

# Define independent reasoning steps
pipeline.add_step("Fetch data from the knowledge base.")
pipeline.add_step("Analyze the input query.")
pipeline.add_step("Generate a summary of related concepts.")

# Execute the pipeline
response = pipeline.execute(agent.model_manager)
print(f"Final Output using Parallel Reasoning: {response}")
