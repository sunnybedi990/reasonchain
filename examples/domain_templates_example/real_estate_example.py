from reasonchain.domain_templates.real_estate import (
    property_valuation,
    buyer_profile_matching,
    rental_yield_prediction,
    market_trend_analysis,
    property_investment_analysis
)

# Use case: Evaluate property value
valuation = property_valuation()
print("Property Valuation:\n", valuation)

# Use case: Match properties to buyer profiles
buyer_match = buyer_profile_matching()
print("\nBuyer Profile Matching:\n", buyer_match)

# Use case: Predict rental yield
rental_yield = rental_yield_prediction()
print("\nRental Yield Prediction:\n", rental_yield)

# Use case: Analyze market trends
market_trends = market_trend_analysis()
print("\nMarket Trend Analysis:\n", market_trends)

# Use case: Evaluate property investment ROI
investment_analysis = property_investment_analysis()
print("\nProperty Investment Analysis:\n", investment_analysis)
