from reasonchain.agent import Agent, MultiAgentSystem
from reasonchain.cot_pipeline import CoTPipeline, TreeOfThoughtPipeline, ParallelCoTPipeline, HybridCoTPipeline
from reasonchain.rag.vector.add_to_vector_db import add_pdf_to_vector_db
from reasonchain.rag.rag_main import query_vector_db

# HEALTHCARE DOMAIN TEMPLATES

def diagnosis_assistance():
    """Assist in medical diagnosis using Chain of Thought reasoning."""
    agent = Agent(name="DiagnosisBot", model_name="gpt-4", api="openai")
    pipeline = CoTPipeline(agent)
    pipeline.add_step("Understand the patient's symptoms.")
    pipeline.add_step("Review relevant medical history and tests.")
    pipeline.add_step("Consult medical literature for similar cases.")
    pipeline.add_step("Generate a potential diagnosis based on gathered information.")
    return pipeline.execute(agent.model_manager)

def treatment_recommendation():
    """Generate treatment recommendations using Tree of Thought reasoning."""
    agent = Agent(name="TreatmentBot", model_name="llama3.1", api="ollama")
    pipeline = TreeOfThoughtPipeline(agent)
    pipeline.add_step("Identify the medical condition being treated.")
    pipeline.add_step("Consider treatment options based on evidence and guidelines.")
    pipeline.add_step("Evaluate potential risks and benefits of each option.")
    pipeline.add_step("Provide the most suitable treatment recommendation.")
    return pipeline.execute(agent.model_manager)

def medical_article_summarization(pdf_path):
    """Summarize medical articles using RAG integration."""
    vector_db_path = "medical_vector_db.index"
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
        query="Summarize key findings from the article.",
        embedding_provider="sentence_transformers",
        embedding_model="all-mpnet-base-v2",
        top_k=5
    )

def patient_progress_tracking():
    """Track and analyze patient progress using Parallel Reasoning."""
    agent = Agent(name="ProgressBot", model_name="gpt-4", api="openai")
    pipeline = ParallelCoTPipeline(agent)
    pipeline.add_step("Review patient health metrics over time.")
    pipeline.add_step("Assess improvements or deteriorations in condition.")
    pipeline.add_step("Recommend adjustments to treatment based on progress.")
    return pipeline.execute(agent.model_manager)

def collaborative_case_review():
    """Collaborate on complex medical cases using Hybrid Reasoning."""
    multi_agent_system = MultiAgentSystem()
    agent1 = Agent(name="DoctorAlpha", model_name="gpt-4", api="openai")
    agent2 = Agent(name="DoctorBeta", model_name="llama3.1", api="ollama")
    multi_agent_system.register_agent(agent1)
    multi_agent_system.register_agent(agent2)

    pipeline = HybridCoTPipeline(agent1)
    pipeline.add_step("Review medical case details.", complexity="low")
    pipeline.add_step("Consult medical literature and collaborate on diagnosis.", complexity="medium")
    pipeline.add_step("Provide treatment options and prognosis.", complexity="high")
    return pipeline.execute(agent1.model_manager)

# Add additional healthcare reasoning templates as needed.
