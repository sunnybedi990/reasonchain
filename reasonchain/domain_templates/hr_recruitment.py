from reasonchain.agent import Agent
from reasonchain.cot_pipeline import CoTPipeline, TreeOfThoughtPipeline, ParallelCoTPipeline, HybridCoTPipeline

def candidate_shortlisting():
    """Shortlist candidates using CoT."""
    agent = Agent(name="RecruitBot", model_name="gpt-4", api="openai")
    pipeline = CoTPipeline(agent)
    pipeline.add_step("Review candidate resumes and profiles.")
    pipeline.add_step("Evaluate skills and experience against job requirements.")
    pipeline.add_step("Recommend top candidates for further review.")
    return pipeline.execute(agent.model_manager)

def team_building_strategy():
    """Explore different team-building strategies using ToT."""
    agent = Agent(name="TeamBot", model_name="llama3.1", api="ollama")
    pipeline = TreeOfThoughtPipeline(agent)
    pipeline.add_step("Assess current team composition and gaps.")
    pipeline.add_step("Propose new team structures and roles.")
    pipeline.add_step("Evaluate the impact of proposed changes.")
    return pipeline.execute(agent.model_manager)

def resume_comparison():
    """Compare multiple resumes using Parallel Reasoning."""
    agent = Agent(name="ResumeBot", model_name="gpt-4", api="openai")
    pipeline = ParallelCoTPipeline(agent)
    pipeline.add_step("Extract key qualifications from each resume.")
    pipeline.add_step("Compare candidates' skills, experience, and fit.")
    pipeline.add_step("Provide a ranked list of candidates.")
    return pipeline.execute(agent.model_manager)

def hybrid_onboarding():
    """Combine candidate evaluation with onboarding optimization."""
    agent = Agent(name="OnboardBot", model_name="gpt-4", api="openai")
    pipeline = HybridCoTPipeline(agent)
    pipeline.add_step("Evaluate new hire performance.", complexity="low")
    pipeline.add_step("Optimize onboarding processes.", complexity="medium")
    pipeline.add_step("Recommend personalized training programs.", complexity="high")
    return pipeline.execute(agent.model_manager)
