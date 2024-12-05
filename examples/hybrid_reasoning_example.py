from reasonchain import Agent
from reasonchain.cot_pipeline import HybridCoTPipeline
from reasonchain.utils import dynamic_complexity_evaluator
from reasonchain.model_manager import ModelManager

# Initialize an agent with OpenAI GPT
agent = Agent(name="HybridBot", model_name="gpt-4o", api='openai')
# Define query
query = "What are the best practices for optimizing SQL queries?"
# Initialize the Hybrid pipeline
pipeline = HybridCoTPipeline(agent, complexity_evaluator=dynamic_complexity_evaluator)

# Define tasks with the query
task_list = [
    f"Understand the user's question: {query}",
    f"Fetch data from the knowledge base about: {query}",
    f"Generate multiple solutions and evaluate the best one for: {query}"
]

# Add steps dynamically with complexity evaluation
for task in task_list:
    pipeline.add_step(task)


# Execute the pipeline
response = pipeline.execute(agent.model_manager)
def summarize_output(output):
    summarizer = ModelManager(model_name="gpt-4o", api='openai')
    summary = summarizer.summarize(output)
    return summary

final_summary = summarize_output(response)
print(f"Final Summarized Output:\n{final_summary}")
