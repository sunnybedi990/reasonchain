from reasonchain.domain_templates.customer_support import (
    query_resolution,
    ticket_routing,
    customer_sentiment_analysis,
    hybrid_support,
)

# Resolve a customer query
query_result = query_resolution()
print("Query Resolution:")
print(query_result)

# Route a customer support ticket
ticket_result = ticket_routing()
print("\nTicket Routing:")
print(ticket_result)

# Analyze customer sentiment
sentiment_result = customer_sentiment_analysis()
print("\nCustomer Sentiment Analysis:")
print(sentiment_result)

# Perform hybrid support combining sentiment analysis with resolutions
hybrid_result = hybrid_support()
print("\nHybrid Support:")
print(hybrid_result)
