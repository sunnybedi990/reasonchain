from reasonchain.domain_templates.healthcare import (
    patient_diagnosis, 
    treatment_recommendation, 
    medical_research_analysis, 
    resource_allocation, 
    public_health_insights
)

# Example usage
diagnosis = patient_diagnosis("Patient has symptoms of fever, cough, and fatigue.")
treatment_plan = treatment_recommendation(diagnosis)
research_summary = medical_research_analysis("path/to/medical_research.pdf")
resource_plan = resource_allocation()
public_health = public_health_insights()

print(diagnosis)
print(treatment_plan)
print(research_summary)
print(resource_plan)
print(public_health)
