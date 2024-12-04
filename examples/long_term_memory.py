from reasonchain import Agent
from reasonchain.cot_pipeline import CoTPipeline
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize an agent with OpenAI GPT
api_key = os.getenv("OPENAI_API_KEY")

agent = Agent(name="MemoryEnhancedBot", model_name="gpt-4o-mini", api_key=api_key)

# Initialize CoT pipeline
pipeline = CoTPipeline(agent=agent)

# Define reasoning steps
pipeline.add_step("Understand the user's question.")
pipeline.add_step("Retrieve any relevant context from memory.")
pipeline.add_step("Answer the question based on available information.")

# Simulate interaction
input_data = "What are the best practices for optimizing SQL queries?"
observed_data = agent.observe(input_data)
response = agent.reason(pipeline)
final_output = agent.act(response)

print(f"\nFinal Output:\n{final_output}")

# Query the memory
query = "How do I optimize SQL queries?"
results = agent.memory.retrieve_long_term(query)

# Display results
print(f"\nResults for '{query}':")
for text, score in results:
    print(f"Text: {text}, Score: {score}")
