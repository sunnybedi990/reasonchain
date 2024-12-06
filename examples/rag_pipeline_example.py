from reasonchain.memory import Memory
from reasonchain.agent import Agent
from reasonchain.rag.vector.add_to_vector_db import add_pdf_to_vector_db
from reasonchain.rag.rag_main import query_vector_db
from reasonchain.cot_pipeline import TreeOfThoughtPipeline
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Memory and Vector Database
memory = Memory(embedding_provider='sentence_transformers',embedding_model="all-mpnet-base-v2", dimension=768, use_gpu=False)
vector_db_path = "vector_db_sql.index"  # Define the vector DB path
vector_db_type = "faiss"  # Use FAISS for vector storage

# Populate vector database with SQL optimization knowledge
print("\n=== Adding SQL Optimization Knowledge to Vector Database ===")
add_pdf_to_vector_db(
    pdf_path="sql_optimization_guide.pdf",  # Path to a PDF with SQL optimization tips
    db_path=vector_db_path,
    db_type=vector_db_type,
    db_config=None,
    embedding_provider="sentence_transformers",
    embedding_model="all-mpnet-base-v2",
    use_gpu=True
)

# Initialize the agent
api_key = os.getenv("OPENAI_API_KEY")
agent = Agent(name="RAGBot", model_name="gpt-4o", api='openai', memory=memory)

# Initialize the reasoning pipeline
pipeline = TreeOfThoughtPipeline(agent=agent)

# Define reasoning steps
pipeline.add_step("Understand the user's question.")
pipeline.add_step("Retrieve relevant context from memory or vector database.")
pipeline.add_step("Generate an answer based on the retrieved context.")

# Execute reasoning with RAG integration
query = "How do I optimize SQL queries?"
print(f"\n=== Querying Vector Database ===")
retrieved_context = query_vector_db(
    db_path=vector_db_path,
    db_type=vector_db_type,
    query=query,
    embedding_provider="sentence_transformers",
    embedding_model="all-mpnet-base-v2",
    top_k=5
)

# Combine the query with the retrieved context
augmented_query = f"{query}\nRelevant Context: {retrieved_context}"

# Execute the pipeline
response = pipeline.execute(agent.model_manager)
print(f"\nFinal Output:\n{response}")
