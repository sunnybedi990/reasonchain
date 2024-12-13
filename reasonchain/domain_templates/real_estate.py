from reasonchain.agent import Agent
from reasonchain.cot_pipeline import CoTPipeline, TreeOfThoughtPipeline, ParallelCoTPipeline

def property_valuation():
    """Evaluate property value using Chain of Thought reasoning."""
    agent = Agent(name="ValuationBot", model_name="gpt-4", api="openai")
    pipeline = CoTPipeline(agent)
    pipeline.add_step("Collect details about the property (location, size, amenities).")
    pipeline.add_step("Analyze market trends in the area.")
    pipeline.add_step("Estimate the property's value based on the collected data.")
    return pipeline.execute(agent.model_manager)

def buyer_profile_matching():
    """Match properties to buyers using Tree of Thought reasoning."""
    agent = Agent(name="MatchBot", model_name="llama3.1", api="ollama")
    pipeline = TreeOfThoughtPipeline(agent)
    pipeline.add_step("Analyze buyer's preferences and budget.")
    pipeline.add_step("Search for properties meeting the criteria.")
    pipeline.add_step("Rank properties by suitability.")
    return pipeline.execute(agent.model_manager)

def rental_yield_prediction():
    """Predict rental yield for investment properties."""
    agent = Agent(name="RentalBot", model_name="gpt-4", api="openai")
    pipeline = CoTPipeline(agent)
    pipeline.add_step("Analyze historical rental data in the area.")
    pipeline.add_step("Estimate expected rental income.")
    pipeline.add_step("Calculate the rental yield based on property value.")
    return pipeline.execute(agent.model_manager)

def market_trend_analysis():
    """Analyze real estate market trends."""
    agent = Agent(name="MarketTrendBot", model_name="gpt-4", api="openai")
    pipeline = CoTPipeline(agent)
    pipeline.add_step("Collect historical pricing and transaction data.")
    pipeline.add_step("Identify trends in pricing and transaction volume.")
    pipeline.add_step("Provide insights into future market movements.")
    return pipeline.execute(agent.model_manager)

def property_investment_analysis():
    """Evaluate the potential ROI for real estate investments."""
    agent = Agent(name="InvestmentBot", model_name="llama3.1", api="ollama")
    pipeline = ParallelCoTPipeline(agent)
    pipeline.add_step("Analyze purchase price and associated costs.")
    pipeline.add_step("Estimate rental income and potential appreciation.")
    pipeline.add_step("Calculate the expected ROI.")
    return pipeline.execute(agent.model_manager)
