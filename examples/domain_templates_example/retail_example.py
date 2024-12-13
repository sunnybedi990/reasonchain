from reasonchain.domain_templates.retail import (
    inventory_optimization,
    pricing_strategy,
    sales_forecasting,
    customer_loyalty_analysis,
    discount_impact_simulation
)

# Use case: Optimize inventory levels
inventory_plan = inventory_optimization()
print("Inventory Optimization:\n", inventory_plan)

# Use case: Develop pricing strategies
pricing_plan = pricing_strategy()
print("\nPricing Strategy:\n", pricing_plan)

# Use case: Forecast sales
sales_forecast = sales_forecasting()
print("\nSales Forecasting:\n", sales_forecast)

# Use case: Analyze customer loyalty
customer_loyalty = customer_loyalty_analysis()
print("\nCustomer Loyalty Analysis:\n", customer_loyalty)

# Use case: Simulate the impact of discounts
discount_simulation = discount_impact_simulation()
print("\nDiscount Impact Simulation:\n", discount_simulation)
