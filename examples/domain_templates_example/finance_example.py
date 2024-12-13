from reasonchain.domain_templates.finance import (
    portfolio_optimization, 
    financial_risk_analysis, 
    stock_performance_prediction, 
    fraud_detection, 
    tax_strategy_planning
)

# Example usage
portfolio_plan = portfolio_optimization()
risk_analysis = financial_risk_analysis()
stock_prediction = stock_performance_prediction()
fraud_check = fraud_detection("path/to/transaction_data.csv")
tax_plan = tax_strategy_planning()

print(portfolio_plan)
print(risk_analysis)
print(stock_prediction)
print(fraud_check)
print(tax_plan)
