from reasonchain.agent import Agent
from reasonchain.cot_pipeline import CoTPipeline, TreeOfThoughtPipeline, ParallelCoTPipeline, HybridCoTPipeline

def itinerary_planning():
    """Build a travel itinerary using CoT."""
    agent = Agent(name="TravelBot", model_name="gpt-4", api="openai")
    pipeline = CoTPipeline(agent)
    pipeline.add_step("Understand the traveler's preferences and budget.")
    pipeline.add_step("Identify suitable destinations and activities.")
    pipeline.add_step("Create a detailed itinerary with timelines.")
    return pipeline.execute(agent.model_manager)

def vacation_scenarios():
    """Explore vacation scenarios using ToT."""
    agent = Agent(name="VacationBot", model_name="llama3.1", api="ollama")
    pipeline = TreeOfThoughtPipeline(agent)
    pipeline.add_step("Analyze vacation options by budget.")
    pipeline.add_step("Propose unique vacation ideas.")
    pipeline.add_step("Evaluate scenarios for feasibility and appeal.")
    return pipeline.execute(agent.model_manager)

def package_comparison():
    """Compare travel packages using Parallel Reasoning."""
    agent = Agent(name="CompareBot", model_name="gpt-4", api="openai")
    pipeline = ParallelCoTPipeline(agent)
    pipeline.add_step("Collect details of travel packages.")
    pipeline.add_step("Analyze packages by price, inclusions, and reviews.")
    pipeline.add_step("Provide a ranked list of best packages.")
    return pipeline.execute(agent.model_manager)

def hybrid_destination_recommendations():
    """Combine customer reviews with destination suggestions."""
    agent = Agent(name="HybridTravelBot", model_name="gpt-4", api="openai")
    pipeline = HybridCoTPipeline(agent)
    pipeline.add_step("Analyze destination reviews.", complexity="low")
    pipeline.add_step("Incorporate traveler preferences.", complexity="medium")
    pipeline.add_step("Recommend destinations based on insights.", complexity="high")
    return pipeline.execute(agent.model_manager)
