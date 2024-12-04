from reasonchain.memory import Memory
from reasonchain.agent import Agent
from reasonchain.rag_integration import RAGIntegration
from reasonchain.cot_pipeline import TreeOfThoughtPipeline
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize an agent with OpenAI GPT
api_key = os.getenv("OPENAI_API_KEY")
# Initialize memory and RAG integration
memory = Memory(embedding_model="all-mpnet-base-v2",dimension=768, use_gpu=False)
rag = RAGIntegration(memory=memory)

# Prepopulate long-term memory
memory.store_long_term("SQL optimization involves using indexes and reducing joins.")
memory.store_long_term("Avoid SELECT * for better query performance.")
memory.store_long_term("Proper indexing can significantly improve SQL query execution times.")

# Initialize agent and pipeline
agent = Agent(name="RAGBot", model_name='gpt-4o', api_key=api_key, memory=memory)
pipeline = TreeOfThoughtPipeline(agent=agent)

# Define reasoning steps
pipeline.add_step("Understand the user's question.")
pipeline.add_step("Retrieve relevant context from memory.")
pipeline.add_step("Generate an answer based on the retrieved context.")

# Execute reasoning with RAG integration
query = "How do I optimize SQL queries?"
augmented_query = rag.augment_with_context(query)
response = pipeline.execute(agent.model_manager)
print(f"Final Output:\n{response}")
