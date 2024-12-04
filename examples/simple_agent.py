from reasonchain import Agent, CoTPipeline

# Initialize an agent
agent = Agent(name="MemoryBot", model="openai-gpt4o")

# Initialize CoT pipeline
pipeline = CoTPipeline(agent=agent)

# Define reasoning steps
pipeline.add_step("Understand the user's question.")
pipeline.add_step("Break down the question into sub-questions.")
pipeline.add_step("Answer each sub-question.")
pipeline.add_step("Combine answers into a comprehensive response.")

# Simulate interaction
input_data = "How do I optimize my database?"
observed_data = agent.observe(input_data)
response = agent.reason(pipeline)
final_output = agent.act(response)

# Demonstrate memory functionality
print("\nShort-term Memory:", agent.memory.retrieve_short_term())
agent.memory.clear_short_term()
print("Short-term Memory after clearing:", agent.memory.retrieve_short_term())
