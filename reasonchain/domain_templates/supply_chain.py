from reasonchain.agent import Agent
from reasonchain.cot_pipeline import HybridCoTPipeline, CoTPipeline

def logistics_planning():
    """Plan logistics and deliveries using Hybrid Reasoning."""
    agent = Agent(name="LogisticsBot", model_name="gpt-4", api="openai")
    pipeline = HybridCoTPipeline(agent)
    pipeline.add_step("Analyze current order volumes.", complexity="low")
    pipeline.add_step("Optimize delivery routes.", complexity="medium")
    pipeline.add_step("Recommend cost-saving logistics strategies.", complexity="high")
    return pipeline.execute(agent.model_manager)

def supplier_risk_analysis():
    """Assess risks associated with suppliers."""
    agent = Agent(name="RiskBot", model_name="llama3.1", api="ollama")
    pipeline = CoTPipeline(agent)
    pipeline.add_step("Evaluate supplier performance metrics.")
    pipeline.add_step("Analyze dependency on key suppliers.")
    pipeline.add_step("Recommend mitigation strategies for high-risk suppliers.")
    return pipeline.execute(agent.model_manager)

def demand_forecasting():
    """Forecast demand for products."""
    agent = Agent(name="DemandBot", model_name="gpt-4", api="openai")
    pipeline = CoTPipeline(agent)
    pipeline.add_step("Analyze historical demand data.")
    pipeline.add_step("Incorporate external factors (e.g., market trends, weather).")
    pipeline.add_step("Generate demand forecasts for the next quarter.")
    return pipeline.execute(agent.model_manager)

def warehouse_optimization():
    """Optimize warehouse operations."""
    agent = Agent(name="WarehouseBot", model_name="gpt-4", api="openai")
    pipeline = CoTPipeline(agent)
    pipeline.add_step("Analyze current inventory and space utilization.")
    pipeline.add_step("Identify inefficiencies in warehouse operations.")
    pipeline.add_step("Recommend strategies to improve efficiency and reduce costs.")
    return pipeline.execute(agent.model_manager)

def transportation_cost_reduction():
    """Recommend strategies to reduce transportation costs."""
    agent = Agent(name="TransportCostBot", model_name="llama3.1", api="ollama")
    pipeline = CoTPipeline(agent)
    pipeline.add_step("Analyze current transportation costs and routes.")
    pipeline.add_step("Identify inefficiencies or opportunities for consolidation.")
    pipeline.add_step("Recommend cost-saving strategies.")
    return pipeline.execute(agent.model_manager)
