from reasonchain import Agent
from reasonchain.cot_pipeline import ParallelCoTPipeline
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize an agent with OpenAI GPT
api_key = os.getenv("OPENAI_API_KEY")
agent = Agent(name="ParallelBot", model_name="gpt-4o", api_key=api_key)

# Initialize Parallel CoT pipeline
pipeline = ParallelCoTPipeline(agent=agent)

# Define independent reasoning steps
pipeline.add_step("Fetch data from the knowledge base.")
pipeline.add_step("Analyze the input query.")
pipeline.add_step("Generate a summary of related concepts.")

# Execute the pipeline
response = pipeline.execute(agent.model_manager)
print(f"Final Output using Parallel Reasoning: {response}")
