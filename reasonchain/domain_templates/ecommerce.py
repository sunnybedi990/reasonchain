from reasonchain.agent import Agent
from reasonchain.cot_pipeline import CoTPipeline, ParallelCoTPipeline, HybridCoTPipeline

def customer_personalization():
    """Recommend personalized products using Chain of Thought reasoning."""
    agent = Agent(name="PersonalizationBot", model_name="gpt-4", api="openai")
    pipeline = CoTPipeline(agent)
    pipeline.add_step("Gather customer's purchase history and interaction data.")
    pipeline.add_step("Analyze preferences using clustering or similarity matching.")
    pipeline.add_step("Recommend tailored products or services.")
    return pipeline.execute(agent.model_manager)

def ad_campaign_optimization():
    """Optimize advertising campaigns using Parallel Reasoning."""
    agent = Agent(name="AdCampaignBot", model_name="llama3.1", api="ollama")
    pipeline = ParallelCoTPipeline(agent)
    pipeline.add_step("Analyze campaign performance metrics (CTR, conversions).")
    pipeline.add_step("Identify high-performing segments and areas for improvement.")
    pipeline.add_step("Recommend optimized targeting and budget allocation.")
    return pipeline.execute(agent.model_manager)

def inventory_demand_forecasting():
    """Forecast inventory demand using Hybrid Reasoning."""
    agent = Agent(name="DemandForecastBot", model_name="gpt-4", api="openai")
    pipeline = HybridCoTPipeline(agent)
    pipeline.add_step("Collect historical sales and inventory data.", complexity="low")
    pipeline.add_step("Incorporate seasonal trends and external factors (e.g., holidays).", complexity="medium")
    pipeline.add_step("Predict future demand and recommend inventory levels.", complexity="high")
    return pipeline.execute(agent.model_manager)
