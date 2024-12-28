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
pdf_path = "pdfs/tsla-20240930-gen.pdf"  # Path to the Tesla Q-10 report
vector_db_path = "vector_db_tesla.index"
vector_db_type = "faiss"  # Can be faiss, milvus, etc.

# Populate vector database with SQL optimization knowledge
print("\n=== Adding Tesla Q-10 Report to Vector Database ===")
add_pdf_to_vector_db(
    file_path=pdf_path,
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
query = "What is the revenue growth of Tesla?"
print(f"\n=== Querying Vector Database ===")
retrieved_context = query_vector_db(
    db_path=vector_db_path,
    db_type=vector_db_type,
    query=query,
    embedding_provider="sentence_transformers",
    embedding_model="all-mpnet-base-v2",
    top_k=5,
    use_gpu=False,
    db_config=None
)
# Access results and metadata
results = retrieved_context["results"]
metadata = retrieved_context["metadata"]

for result in results:
    print(f"Text: {result['text']}")
    print(f"Score: {result['score']}")
    print(f"Chunk Metadata: {result['metadata']}")
    print("---")

print(f"Query Metadata: {metadata}")# Combine the query with the retrieved context
# augmented_query = f"{query}\nRelevant Context: {retrieved_context}"

# # Execute the pipeline
# response = pipeline.execute(agent.model_manager)
# print(f"\nFinal Output:\n{response}")
