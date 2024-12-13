from reasonchain.domain_templates.education import (
    curriculum_design, 
    student_progress_evaluation, 
    educational_article_summarization, 
    learning_path_recommendation, 
    collaborative_learning
)

# Example usage
curriculum_plan = curriculum_design()
evaluation = student_progress_evaluation()
article_summary = educational_article_summarization("path/to/educational_article.pdf")
learning_path = learning_path_recommendation()
collaboration_results = collaborative_learning()

print(curriculum_plan)
print(evaluation)
print(article_summary)
print(learning_path)
print(collaboration_results)
