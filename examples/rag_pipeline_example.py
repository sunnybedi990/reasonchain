from reasonchain.memory import Memory
from reasonchain.agent import Agent
from reasonchain.rag.vector.add_to_vector_db import add_data_to_vector_db
from reasonchain.rag.rag_main import query_vector_db
from reasonchain.cot_pipeline import TreeOfThoughtPipeline
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Memory and Vector Database
memory = Memory(embedding_provider='sentence_transformers',embedding_model="all-mpnet-base-v2", dimension=768, use_gpu=False)
file_paths = [
    "pdfs/tsla-20240930-gen.pdf",
    # Add more PDF paths here
]  # List of files to process
vector_db_path = "vector_db_tesla.index"
vector_db_type = "faiss"  # Can be faiss, milvus, etc.
# db_config = {
#     "mode": "cloud",
#     "api_key": os.getenv("QDRANT_API_KEY"),
#     "host": os.getenv("QDRANT_HOST"),
#     "port": os.getenv("QDRANT_PORT"),
#     "collection_name": os.getenv("QDRANT_COLLECTION_NAME")
# }

# db_config = {
#     "mode": "cloud",
#     "api_key": os.getenv("PINECONE_API_KEY"),
#     "index_name": os.getenv("PINECONE_INDEX_NAME")
# }

db_config = {
    "mode" : 'cloud',
    "class_name" : "tesla_q10",
    "api_key" : os.getenv("WEAVIATE_API_KEY"),
    "WEAVIATE_CLUSTER_URL" : os.getenv("WEAVIATE_CLUSTER_URL")
}


# Populate vector database with SQL optimization knowledge
print("\n=== Adding Tesla Reports to Vector Database ===")
result = add_data_to_vector_db(
    file_paths=file_paths,  # Now passing a list of files
    db_path=vector_db_path,
    db_type=vector_db_type,
    db_config=db_config,
    embedding_provider="sentence_transformers",
    embedding_model="all-mpnet-base-v2",
    use_gpu=True
)
print(result)
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
    db_config=db_config
)
# Access results and metadata
results = retrieved_context["results"]
metadata = retrieved_context["metadata"]

for result in results:
    print(result)
    print(f"Text: {result['text']}")
    print(f"Score: {result['score']}")
    print(f"Chunk Metadata: {result['metadata']}")
    print("---")

print(f"Query Metadata: {metadata}")# Combine the query with the retrieved context
# augmented_query = f"{query}\nRelevant Context: {retrieved_context}"

# # Execute the pipeline
# response = pipeline.execute(agent.model_manager)
# print(f"\nFinal Output:\n{response}")
