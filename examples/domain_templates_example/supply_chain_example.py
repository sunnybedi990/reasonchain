from reasonchain.domain_templates.supply_chain import (
    logistics_planning,
    supplier_risk_analysis,
    demand_forecasting,
    warehouse_optimization,
    transportation_cost_reduction
)

# Use case: Plan logistics and deliveries
logistics_plan = logistics_planning()
print("Logistics Planning:\n", logistics_plan)

# Use case: Assess supplier risks
risk_analysis = supplier_risk_analysis()
print("\nSupplier Risk Analysis:\n", risk_analysis)

# Use case: Forecast product demand
demand_forecast = demand_forecasting()
print("\nDemand Forecasting:\n", demand_forecast)

# Use case: Optimize warehouse operations
warehouse_plan = warehouse_optimization()
print("\nWarehouse Optimization:\n", warehouse_plan)

# Use case: Reduce transportation costs
transportation_costs = transportation_cost_reduction()
print("\nTransportation Cost Reduction:\n", transportation_costs)
