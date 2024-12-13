from reasonchain.domain_templates.legal import (
    case_summary_generation, 
    legal_document_analysis, 
    compliance_check, 
    dispute_resolution, 
    contract_review
)

# Example usage
case_summary = case_summary_generation("path/to/case_details.pdf")
document_analysis = legal_document_analysis("path/to/legal_document.pdf")
compliance_report = compliance_check("path/to/company_policies.pdf")
dispute_suggestions = dispute_resolution("path/to/dispute_details.txt")
contract_review_feedback = contract_review("path/to/contract_document.pdf")

print(case_summary)
print(document_analysis)
print(compliance_report)
print(dispute_suggestions)
print(contract_review_feedback)
