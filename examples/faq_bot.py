from reasonchain import Agent, CoTPipeline
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize an agent with OpenAI GPT
api_key = os.getenv("OPENAI_API_KEY")
agent = Agent(name="FAQBot", model_name="gpt-4o-mini", api_key=api_key)

# Initialize CoT pipeline
pipeline = CoTPipeline(agent=agent)

# Define reasoning steps
pipeline.add_step("Understand the user's question.")
pipeline.add_step("Break down the question into sub-questions.")
pipeline.add_step("Answer each sub-question.")
pipeline.add_step("Combine answers into a comprehensive response.")

# Simulate interaction
input_data = "How do I optimize a SQL database?"
observed_data = agent.observe(input_data)
response = agent.reason(pipeline)
final_output = agent.act(response)

print(f"\nFinal Output:\n{final_output}")
