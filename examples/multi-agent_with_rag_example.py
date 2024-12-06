from reasonchain.agent import Agent, MultiAgentSystem
from reasonchain.memory import Memory
from reasonchain.utils import (
    assign_and_execute_task,
    store_in_shared_memory,
    retrieve_from_shared_memory,
    collaborate_on_task
)
from reasonchain.rag.vector.add_to_vector_db import add_pdf_to_vector_db
from reasonchain.rag.rag_main import query_vector_db

# Initialize the Multi-Agent System
multi_agent_system = MultiAgentSystem()

# Create agents with different roles and models
agent1 = Agent(name="AgentAlpha", role="extractor", model_name="gpt-4o", api="openai")
agent2 = Agent(name="AgentBeta", role="analyst", model_name="llama3.1:latest", api="ollama")
agent3 = Agent(name="AgentGamma", role="summarizer", model_name="llama-3.1-8b-instant", api="groq")

# Register agents in the system
multi_agent_system.register_agent(agent1)
multi_agent_system.register_agent(agent2)
multi_agent_system.register_agent(agent3)

# Step 1: Add the Tesla Q-10 report to the vector database
pdf_path = "pdfs/tsla-20240930-gen.pdf"  # Path to the Tesla Q-10 report
vector_db_path = "vector_db_tesla.index"
vector_db_type = "faiss"  # Can be faiss, milvus, etc.
print("\n=== Adding Tesla Q-10 Report to Vector Database ===")
add_pdf_to_vector_db(
    pdf_path=pdf_path,
    db_path=vector_db_path,
    db_type=vector_db_type,
    db_config=None,
    embedding_provider="sentence_transformers",
    embedding_model="all-mpnet-base-v2",
    use_gpu=True
)

# Define a shared task
shared_task = {
    "name": "Analyze Tesla Q-10 Report",
    "description": "Extract key financial highlights, analyze trends, and summarize insights from the Tesla Q-10 report.",
    "priority": 5
}

# Add the task to the system
multi_agent_system.add_task(shared_task)

# Step 2: Assign and execute task for AgentAlpha (Extractor)
multi_agent_system.assign_task()
query = "Extract financial highlights from the Tesla Q-10 report."
response_alpha = query_vector_db(
    db_path=vector_db_path,
    db_type=vector_db_type,
    query=query,
    db_config=None,
    embedding_provider="sentence_transformers",
    embedding_model="all-mpnet-base-v2",
    top_k=20,
    
)

# Store the extracted financial highlights in shared memory
store_in_shared_memory(agent1.shared_memory, "financial_highlights", response_alpha)

# Step 3: Collaboration setup for AgentBeta and AgentGamma
task_description = "Analyze trends and summarize insights from Tesla's Q-10 report."
# Collaborate on the task
successful_agents = collaborate_on_task(
    multi_agent_system, 
    ["AgentBeta", "AgentGamma"], 
    task_description
)

# Display successful agents
print("\n=== Successful Collaboration ===")
print(f"Agents involved: {', '.join(successful_agents)}")

# Step 4: AgentBeta analyzes the financial highlights
financial_highlights = retrieve_from_shared_memory(agent2.shared_memory, "financial_highlights")
if financial_highlights:
    analysis_query = f"Analyze trends and provide insights based on the following financial highlights: {financial_highlights}"
    response_beta = assign_and_execute_task(agent2, analysis_query)
    store_in_shared_memory(agent2.shared_memory, "analysis", response_beta)
else:
    response_beta = None

# Step 5: AgentGamma summarizes the analysis
analysis_context = retrieve_from_shared_memory(agent3.shared_memory, "analysis")
if analysis_context:
    summary_query = f"Summarize the analysis into key insights for Tesla's Q-10 report: {analysis_context}"
    response_gamma = assign_and_execute_task(agent3, summary_query)
else:
    response_gamma = None

# Step 6: Display results
print("\n=== Final Collaboration Results ===")
print(f"AgentAlpha (Extractor): {response_alpha}")
print(f"AgentBeta (Analyst): {response_beta}")
print(f"AgentGamma (Summarizer): {response_gamma}")
