from reasonchain.agent import Agent, MultiAgentSystem
from reasonchain.cot_pipeline import CoTPipeline, TreeOfThoughtPipeline, ParallelCoTPipeline, HybridCoTPipeline
from reasonchain.rag.vector.add_to_vector_db import add_pdf_to_vector_db
from reasonchain.rag.rag_main import query_vector_db

# EDUCATION DOMAIN TEMPLATES

def curriculum_design():
    """Design a curriculum using Chain of Thought reasoning."""
    agent = Agent(name="CurriculumBot", model_name="gpt-4", api="openai")
    pipeline = CoTPipeline(agent)
    pipeline.add_step("Understand the educational goals and target audience.")
    pipeline.add_step("Identify key topics and learning objectives.")
    pipeline.add_step("Design the curriculum structure and resources.")
    pipeline.add_step("Recommend teaching methods and assessment strategies.")
    return pipeline.execute(agent.model_manager)

def student_progress_evaluation():
    """Evaluate student progress using Tree of Thought reasoning."""
    agent = Agent(name="EvaluationBot", model_name="llama3.1", api="ollama")
    pipeline = TreeOfThoughtPipeline(agent)
    pipeline.add_step("Review student performance data (grades, assignments, etc.).")
    pipeline.add_step("Identify areas of strength and weakness.")
    pipeline.add_step("Provide feedback and suggestions for improvement.")
    pipeline.add_step("Recommend personalized learning resources or interventions.")
    return pipeline.execute(agent.model_manager)

def educational_article_summarization(pdf_path):
    """Summarize educational articles using RAG integration."""
    vector_db_path = "education_vector_db.index"
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
        query="Summarize the key takeaways from this educational article.",
        embedding_provider="sentence_transformers",
        embedding_model="all-mpnet-base-v2",
        top_k=5
    )

def learning_path_recommendation():
    """Recommend learning paths for students using Parallel Reasoning."""
    agent = Agent(name="PathBot", model_name="gpt-4", api="openai")
    pipeline = ParallelCoTPipeline(agent)
    pipeline.add_step("Assess student's current skills and knowledge.")
    pipeline.add_step("Evaluate career goals and learning preferences.")
    pipeline.add_step("Recommend an optimal learning path and resources.")
    return pipeline.execute(agent.model_manager)

def collaborative_learning():
    """Enable collaborative learning using Hybrid Reasoning."""
    multi_agent_system = MultiAgentSystem()
    agent1 = Agent(name="LearnerAlpha", model_name="gpt-4", api="openai")
    agent2 = Agent(name="LearnerBeta", model_name="llama3.1", api="ollama")
    agent3 = Agent(name="FacilitatorGamma", model_name="gpt-4", api="openai")

    # Register agents
    multi_agent_system.register_agent(agent1)
    multi_agent_system.register_agent(agent2)
    multi_agent_system.register_agent(agent3)

    # Define a shared task
    shared_task = {
        "name": "Collaborative Learning Project",
        "description": "Analyze and solve a complex educational problem collaboratively.",
        "priority": 5
    }

    # Add the task to the system
    multi_agent_system.add_task(shared_task)

    # Assign and execute task
    multi_agent_system.assign_task()
    agent1_response = agent1.observe("Analyze the problem statement and identify key challenges.")
    agent2_response = agent2.observe(f"Expand on AgentAlpha's analysis: {agent1_response}. Provide additional insights.")
    agent3_response = agent3.observe(f"Summarize findings from both agents and propose actionable solutions.")

    # Return collaboration results
    return {
        "AgentAlpha": agent1_response,
        "AgentBeta": agent2_response,
        "FacilitatorGamma": agent3_response
    }

