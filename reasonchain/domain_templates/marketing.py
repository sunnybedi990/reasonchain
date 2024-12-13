from reasonchain.agent import Agent
from reasonchain.cot_pipeline import ParallelCoTPipeline, CoTPipeline, TreeOfThoughtPipeline

def campaign_performance_analysis():
    """Evaluate and optimize marketing campaign performance using Parallel Reasoning."""
    agent = Agent(name="CampaignPerformanceBot", model_name="gpt-4", api="openai")
    pipeline = ParallelCoTPipeline(agent)
    pipeline.add_step("Collect performance metrics (CTR, conversions, ROI).")
    pipeline.add_step("Identify high-performing and underperforming segments.")
    pipeline.add_step("Recommend adjustments for improved performance.")
    return pipeline.execute(agent.model_manager)

def audience_segmentation():
    """Segment target audiences using Tree of Thought reasoning."""
    agent = Agent(name="AudienceSegmentationBot", model_name="llama3.1", api="ollama")
    pipeline = TreeOfThoughtPipeline(agent)
    pipeline.add_step("Analyze customer demographics and behavior.")
    pipeline.add_step("Cluster similar audience segments.")
    pipeline.add_step("Propose tailored marketing strategies for each segment.")
    return pipeline.execute(agent.model_manager)

def content_strategy():
    """Develop a content strategy using Chain of Thought reasoning."""
    agent = Agent(name="ContentStrategyBot", model_name="gpt-4", api="openai")
    pipeline = CoTPipeline(agent)
    pipeline.add_step("Analyze audience interests and engagement metrics.")
    pipeline.add_step("Identify trending topics and content gaps.")
    pipeline.add_step("Recommend content types and schedules.")
    return pipeline.execute(agent.model_manager)
