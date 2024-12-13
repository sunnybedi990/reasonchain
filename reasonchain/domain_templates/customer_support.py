from reasonchain.agent import Agent
from reasonchain.cot_pipeline import CoTPipeline, TreeOfThoughtPipeline, ParallelCoTPipeline, HybridCoTPipeline

def query_resolution():
    """Resolve customer issues step by step using CoT."""
    agent = Agent(name="QueryBot", model_name="gpt-4", api="openai")
    pipeline = CoTPipeline(agent)
    pipeline.add_step("Understand the customer's query.")
    pipeline.add_step("Check relevant knowledge bases for solutions.")
    pipeline.add_step("Provide a detailed resolution.")
    return pipeline.execute(agent.model_manager)

def ticket_routing():
    """Route customer tickets based on complexity using Tree of Thought."""
    agent = Agent(name="TicketBot", model_name="llama3.1", api="ollama")
    pipeline = TreeOfThoughtPipeline(agent)
    pipeline.add_step("Categorize the ticket by urgency and complexity.")
    pipeline.add_step("Assign the ticket to the appropriate team.")
    pipeline.add_step("Notify the customer of the ticket status.")
    return pipeline.execute(agent.model_manager)

def customer_sentiment_analysis():
    """Analyze customer sentiment using Parallel Reasoning."""
    agent = Agent(name="SentimentBot", model_name="gpt-4", api="openai")
    pipeline = ParallelCoTPipeline(agent)
    pipeline.add_step("Extract customer sentiment from their messages.")
    pipeline.add_step("Classify sentiment as positive, neutral, or negative.")
    pipeline.add_step("Provide sentiment trends for decision-making.")
    return pipeline.execute(agent.model_manager)

def hybrid_support():
    """Combine sentiment analysis with resolution suggestions."""
    agent = Agent(name="SupportBot", model_name="gpt-4", api="openai")
    pipeline = HybridCoTPipeline(agent)
    pipeline.add_step("Analyze customer sentiment.", complexity="low")
    pipeline.add_step("Retrieve possible resolutions.", complexity="medium")
    pipeline.add_step("Provide resolution suggestions based on sentiment.", complexity="high")
    return pipeline.execute(agent.model_manager)
