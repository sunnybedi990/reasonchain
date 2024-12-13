from reasonchain.domain_templates.travel_hospitality import (
    itinerary_planning,
    vacation_scenarios,
    package_comparison,
    hybrid_destination_recommendations,
)

# Plan a travel itinerary
itinerary_result = itinerary_planning()
print("Itinerary Planning:")
print(itinerary_result)

# Explore different vacation scenarios
vacation_result = vacation_scenarios()
print("\nVacation Scenarios:")
print(vacation_result)

# Compare travel packages
package_result = package_comparison()
print("\nPackage Comparison:")
print(package_result)

# Recommend destinations combining customer reviews and preferences
hybrid_destination_result = hybrid_destination_recommendations()
print("\nHybrid Destination Recommendations:")
print(hybrid_destination_result)
