from reasonchain.agent import Agent
from reasonchain.cot_pipeline import TreeOfThoughtPipeline, CoTPipeline, ParallelCoTPipeline

def renewable_energy_forecasting():
    """Forecast renewable energy production using Tree of Thought reasoning."""
    agent = Agent(name="EnergyForecastBot", model_name="gpt-4", api="openai")
    pipeline = TreeOfThoughtPipeline(agent)
    pipeline.add_step("Collect weather data and historical energy output.")
    pipeline.add_step("Analyze trends in renewable energy production.")
    pipeline.add_step("Forecast energy generation for upcoming periods.")
    return pipeline.execute(agent.model_manager)

def grid_efficiency_optimization():
    """Optimize power grid efficiency using Parallel Reasoning."""
    agent = Agent(name="GridEfficiencyBot", model_name="llama3.1", api="ollama")
    pipeline = ParallelCoTPipeline(agent)
    pipeline.add_step("Analyze power grid loads and inefficiencies.")
    pipeline.add_step("Recommend strategies for load balancing.")
    pipeline.add_step("Propose energy-saving measures for peak periods.")
    return pipeline.execute(agent.model_manager)

def carbon_footprint_analysis():
    """Perform carbon footprint analysis using Chain of Thought reasoning."""
    agent = Agent(name="CarbonAnalysisBot", model_name="gpt-4", api="openai")
    pipeline = CoTPipeline(agent)
    pipeline.add_step("Gather data on energy consumption and sources.")
    pipeline.add_step("Analyze emissions per energy source.")
    pipeline.add_step("Provide actionable recommendations to reduce carbon footprint.")
    return pipeline.execute(agent.model_manager)
