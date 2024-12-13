from reasonchain.agent import Agent
from reasonchain.cot_pipeline import CoTPipeline, HybridCoTPipeline

def traffic_prediction():
    """Predict traffic conditions using Hybrid Reasoning."""
    agent = Agent(name="TrafficPredictBot", model_name="gpt-4", api="openai")
    pipeline = HybridCoTPipeline(agent)
    pipeline.add_step("Collect historical traffic data.", complexity="low")
    pipeline.add_step("Analyze real-time road and weather conditions.", complexity="medium")
    pipeline.add_step("Forecast traffic patterns and suggest alternative routes.", complexity="high")
    return pipeline.execute(agent.model_manager)

def fleet_optimization():
    """Optimize fleet operations using Chain of Thought reasoning."""
    agent = Agent(name="FleetOptimizerBot", model_name="llama3.1", api="ollama")
    pipeline = CoTPipeline(agent)
    pipeline.add_step("Analyze fleet utilization and maintenance schedules.")
    pipeline.add_step("Identify inefficiencies and cost-saving opportunities.")
    pipeline.add_step("Recommend strategies to improve fleet efficiency.")
    return pipeline.execute(agent.model_manager)

def public_transport_scheduling():
    """Optimize public transport schedules."""
    agent = Agent(name="TransportSchedulerBot", model_name="gpt-4", api="openai")
    pipeline = CoTPipeline(agent)
    pipeline.add_step("Analyze passenger demand patterns.")
    pipeline.add_step("Incorporate traffic and peak-hour data.")
    pipeline.add_step("Recommend optimized schedules for public transport services.")
    return pipeline.execute(agent.model_manager)
