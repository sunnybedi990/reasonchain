from reasonchain.agent import Agent, MultiAgentSystem
from reasonchain.cot_pipeline import CoTPipeline, TreeOfThoughtPipeline, ParallelCoTPipeline, HybridCoTPipeline
from reasonchain.rag.vector.add_to_vector_db import add_pdf_to_vector_db
from reasonchain.rag.rag_main import query_vector_db

# LEGAL DOMAIN TEMPLATES

def case_analysis():
    """Analyze a legal case using Chain of Thought reasoning."""
    agent = Agent(name="LegalBot", model_name="gpt-4", api="openai")
    pipeline = CoTPipeline(agent)
    pipeline.add_step("Understand the legal issue in the case.")
    pipeline.add_step("Identify relevant laws and precedents.")
    pipeline.add_step("Analyze the facts of the case.")
    pipeline.add_step("Provide a legal argument or solution.")
    return pipeline.execute(agent.model_manager)

def contract_summarization():
    """Summarize a legal contract using Tree of Thought reasoning."""
    agent = Agent(name="ContractBot", model_name="llama3.1", api="ollama")
    pipeline = TreeOfThoughtPipeline(agent)
    pipeline.add_step("Break down the contract into clauses.")
    pipeline.add_step("Identify key responsibilities and obligations.")
    pipeline.add_step("Summarize risks and benefits.")
    pipeline.add_step("Provide an overview of compliance requirements.")
    return pipeline.execute(agent.model_manager)

def precedent_retrieval(pdf_path, query):
    """Retrieve legal precedents using RAG."""
    vector_db_path = "legal_vector_db.index"
    add_pdf_to_vector_db(
        pdf_path=pdf_path,
        db_path=vector_db_path,
        db_type="faiss",
        embedding_provider="sentence_transformers",
        embedding_model="all-mpnet-base-v2",
        use_gpu=True
    )
    return query_vector_db(
        db_path=vector_db_path,
        db_type="faiss",
        query=query,
        embedding_provider="sentence_transformers",
        embedding_model="all-mpnet-base-v2",
        top_k=5
    )

def compliance_check():
    """Perform compliance checks using Parallel Reasoning."""
    agent = Agent(name="ComplianceBot", model_name="gpt-4", api="openai")
    pipeline = ParallelCoTPipeline(agent)
    pipeline.add_step("Check if the document meets GDPR regulations.")
    pipeline.add_step("Ensure compliance with HIPAA for sensitive data.")
    pipeline.add_step("Review for adherence to internal policies.")
    return pipeline.execute(agent.model_manager)

def complex_case_collaboration():
    """Collaborate on a complex legal case using Hybrid Reasoning."""
    multi_agent_system = MultiAgentSystem()
    agent1 = Agent(name="LegalAlpha", model_name="gpt-4", api="openai")
    agent2 = Agent(name="LegalBeta", model_name="llama3.1", api="ollama")
    multi_agent_system.register_agent(agent1)
    multi_agent_system.register_agent(agent2)

    pipeline = HybridCoTPipeline(agent1)
    pipeline.add_step("Understand the case.", complexity="low")
    pipeline.add_step("Retrieve relevant precedents.", complexity="medium")
    pipeline.add_step("Evaluate legal arguments and propose solutions.", complexity="high")
    return pipeline.execute(agent1.model_manager)

# Add additional legal reasoning templates as needed.
