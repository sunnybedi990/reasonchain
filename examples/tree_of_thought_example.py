from reasonchain import Agent
from reasonchain.cot_pipeline import TreeOfThoughtPipeline

agent = Agent(name="ToTBot", model_name="gpt-4o", api='openai')

# Initialize ToT pipeline
pipeline = TreeOfThoughtPipeline(agent=agent)

# Define reasoning steps
pipeline.add_step("Understand the user's question.")
pipeline.add_step("Generate multiple possible solutions.")
pipeline.add_step("Evaluate the solutions for the best one.")

# Execute the pipeline
response = pipeline.execute(agent.model_manager)
print(f"Final Output using ToT:\n{response}")
