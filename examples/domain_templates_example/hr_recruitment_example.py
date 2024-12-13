from reasonchain.domain_templates.hr_recruitment import (
    candidate_shortlisting,
    team_building_strategy,
    resume_comparison,
    hybrid_onboarding,
)

# Shortlist candidates for a job
shortlist_result = candidate_shortlisting()
print("Candidate Shortlisting:")
print(shortlist_result)

# Explore team-building strategies
team_strategy_result = team_building_strategy()
print("\nTeam Building Strategy:")
print(team_strategy_result)

# Compare multiple resumes
resume_result = resume_comparison()
print("\nResume Comparison:")
print(resume_result)

# Perform hybrid onboarding combining evaluation with training
onboarding_result = hybrid_onboarding()
print("\nHybrid Onboarding:")
print(onboarding_result)
