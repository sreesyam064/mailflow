"""
Deterministic department routing

No ML model here by design: routing must be explainable, auditable, and instant.
A dict lookup is the correct tool, not a classifier.
"""

CATEGORY_TO_DEPARTMENT = {
    "Billing": "Finance",
    "Technical Support": "Technical Support Team",
    "Sales": "Sales",
    "General Inquiry": "General Support",
}

_EXPLANATIONS = {
    "Billing": "Billing-related emails are routed to Finance for payment/invoice handling.",
    "Technical Support": "Technical issues are routed to the Technical Support team for troubleshooting.",
    "Sales": "Sales inquiries are routed to the Sales team for pricing/plan discussions.",
    "General Inquiry": "General or miscellaneous requests are routed to General Support for triage.",
}


def route(category: str) -> dict:
    # Returns {'department': str, 'explanation': str}. 
    # Unknown categories fall back to General Support rather than raising, since the API must
    # never crash on an unexpected model output.
    department = CATEGORY_TO_DEPARTMENT.get(category, "General Support")
    explanation = _EXPLANATIONS.get(
        category, f"Category '{category}' is not in the routing table; defaulted to General Support."
    )
    return {"department": department, "explanation": explanation}

