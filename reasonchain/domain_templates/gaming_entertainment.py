from reasonchain.agent import Agent
from reasonchain.cot_pipeline import CoTPipeline, TreeOfThoughtPipeline, ParallelCoTPipeline, HybridCoTPipeline

def storyline_generation():
    """Develop a game storyline using CoT."""
    agent = Agent(name="StoryBot", model_name="gpt-4", api="openai")
    pipeline = CoTPipeline(agent)
    pipeline.add_step("Define the main plot and characters.")
    pipeline.add_step("Add conflicts and challenges to the storyline.")
    pipeline.add_step("Create an engaging resolution.")
    return pipeline.execute(agent.model_manager)

def gameplay_mechanics():
    """Explore multiple gameplay mechanics using ToT."""
    agent = Agent(name="MechanicsBot", model_name="llama3.1", api="ollama")
    pipeline = TreeOfThoughtPipeline(agent)
    pipeline.add_step("List existing gameplay mechanics.")
    pipeline.add_step("Propose innovative gameplay ideas.")
    pipeline.add_step("Evaluate mechanics for player engagement.")
    return pipeline.execute(agent.model_manager)

def player_feedback_analysis():
    """Analyze player feedback across levels."""
    agent = Agent(name="FeedbackBot", model_name="gpt-4", api="openai")
    pipeline = ParallelCoTPipeline(agent)
    pipeline.add_step("Collect feedback from different levels.")
    pipeline.add_step("Identify common pain points or improvements.")
    pipeline.add_step("Provide actionable insights for game designers.")
    return pipeline.execute(agent.model_manager)

def hybrid_storyline_development():
    """Combine gameplay metrics with storyline development."""
    agent = Agent(name="HybridStoryBot", model_name="gpt-4", api="openai")
    pipeline = HybridCoTPipeline(agent)
    pipeline.add_step("Analyze gameplay data.", complexity="low")
    pipeline.add_step("Incorporate player preferences.", complexity="medium")
    pipeline.add_step("Refine storyline based on insights.", complexity="high")
    return pipeline.execute(agent.model_manager)
