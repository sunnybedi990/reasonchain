from reasonchain.agent import Agent, MultiAgentSystem
from reasonchain.cot_pipeline import CoTPipeline, TreeOfThoughtPipeline, ParallelCoTPipeline, HybridCoTPipeline
from reasonchain.rag.vector.add_to_vector_db import add_pdf_to_vector_db
from reasonchain.rag.rag_main import query_vector_db

# FINANCE DOMAIN TEMPLATES

def financial_analysis():
    """Perform financial analysis using Chain of Thought reasoning."""
    agent = Agent(name="FinanceBot", model_name="gpt-4", api="openai")
    pipeline = CoTPipeline(agent)
    pipeline.add_step("Understand the financial data provided.")
    pipeline.add_step("Analyze key financial metrics like revenue, profit margins, etc.")
    pipeline.add_step("Identify trends and anomalies in financial performance.")
    pipeline.add_step("Generate insights based on the financial analysis.")
    return pipeline.execute(agent.model_manager)

def investment_advice():
    """Provide investment advice using Tree of Thought reasoning."""
    agent = Agent(name="InvestmentBot", model_name="llama3.1", api="ollama")
    pipeline = TreeOfThoughtPipeline(agent)
    pipeline.add_step("Identify the investor's financial goals.")
    pipeline.add_step("Assess the risk tolerance and market conditions.")
    pipeline.add_step("Recommend investment strategies or assets based on goals and risks.")
    pipeline.add_step("Provide a rationale for the chosen investment strategy.")
    return pipeline.execute(agent.model_manager)

def stock_market_report_summarization(pdf_path):
    """Summarize stock market reports using RAG integration."""
    vector_db_path = "finance_vector_db.index"
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
        query="Summarize the key stock market trends and insights.",
        embedding_provider="sentence_transformers",
        embedding_model="all-mpnet-base-v2",
        top_k=5
    )

def risk_assessment():
    """Assess financial risk using Parallel Reasoning."""
    agent = Agent(name="RiskBot", model_name="gpt-4", api="openai")
    pipeline = ParallelCoTPipeline(agent)
    pipeline.add_step("Evaluate financial risks in the current market.")
    pipeline.add_step("Assess risk in specific investment portfolios.")
    pipeline.add_step("Recommend risk mitigation strategies.")
    return pipeline.execute(agent.model_manager)

def portfolio_management():
    """Manage investment portfolios using Hybrid Reasoning."""
    multi_agent_system = MultiAgentSystem()
    agent1 = Agent(name="PortfolioAlpha", model_name="gpt-4", api="openai")
    agent2 = Agent(name="PortfolioBeta", model_name="llama3.1", api="ollama")
    multi_agent_system.register_agent(agent1)
    multi_agent_system.register_agent(agent2)

    pipeline = HybridCoTPipeline(agent1)
    pipeline.add_step("Review portfolio performance and market trends.", complexity="low")
    pipeline.add_step("Analyze investment diversification and potential risks.", complexity="medium")
    pipeline.add_step("Recommend adjustments to optimize portfolio returns.", complexity="high")
    return pipeline.execute(agent1.model_manager)

# Add additional finance reasoning templates as needed.
