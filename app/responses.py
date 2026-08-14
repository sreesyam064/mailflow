"""
Deterministic, template-based recommended actions and suggested responses.
Intentionally NOT LLM-generated: templates are explanable, reviewable, and free —
appriopriate for a lightweight, low-cost deployment.
"""

_RECOMMENDED_ACTIONS = {
    ("Billing", "HIGH"): "Verify the payment/transaction immediately and investigate for duplicate charges or billing errors before responding.",
    ("Billing", "MEDIUM"): "Review the billing account and confirm transaction details before replying.",
    ("Billing", "LOW"): "Review the billing inquiry during regular queue processing.",
    ("Technical Support", "HIGH"): "Escalate to on-call engineering; investigate the reported issue as a potential service-impacting incident.",
    ("Technical Support", "MEDIUM"): "Assign to a support engineer for troubleshooting within normal SLA.",
    ("Technical Support", "LOW"): "Log the issue and address it in the standard support queue.",
    ("Sales", "HIGH"): "Route to a sales representative for immediate follow-up — high-priority sales opportunity.",
    ("Sales", "MEDIUM"): "Assign to a sales representative for a timely follow-up.",
    ("Sales", "LOW"): "Add to the sales queue for standard follow-up.",
    ("General Inquiry", "HIGH"): "Escalate to a support manager for prompt review.",
    ("General Inquiry", "MEDIUM"): "Assign to General Support for a standard response.",
    ("General Inquiry", "LOW"): "Handle during regular queue processing.",
}

_RESPONSE_TEMPLATES = {
    "Billing": {
        "Negative": "We're sorry for the trouble caused by this billing issue. Our finance team will review the transaction and verify the payment status as a priority, and we'll update you as soon as possible.",
        "Neutral": "Thank you for reaching out about your billing inquiry. Our finance team will review the transaction and get back to you shortly.",
        "Positive": "Thanks for reaching out! We'll take a look at your billing question and follow up shortly.",
    },
    "Technical Support": {
        "Negative": "We're sorry for the disruption this technical issue has caused. Our support team is investigating and will keep you updated as we work toward a resolution.",
        "Neutral": "Thank you for reporting this technical issue. Our support team will investigate and follow up with next steps.",
        "Positive": "Thanks for the report! Our technical team will look into this and get back to you soon.",
    },
    "Sales": {
        "Negative": "Thank you for your patience — we'd like to address your concerns directly. A member of our sales team will reach out shortly to help.",
        "Neutral": "Thank you for your interest! A member of our sales team will follow up shortly with more information.",
        "Positive": "Thanks so much for reaching out — we're excited to help! A sales representative will follow up shortly.",
    },
    "General Inquiry": {
        "Negative": "We're sorry to hear about your experience. A member of our support team will review your message and respond as soon as possible.",
        "Neutral": "Thank you for contacting us. A member of our team will review your message and respond shortly.",
        "Positive": "Thanks for reaching out! We'll review your message and get back to you soon.",
    },
}

_SUBJECT_PREFIX = {
    "Billing": "Re: Billing Inquiry",
    "Technical Support": "Re: Technical Support Request",
    "Sales": "Re: Your Sales Inquiry",
    "General Inquiry": "Re: Your Inquiry",
}


def get_recommended_action(category: str, priority: str) -> str:
    return _RECOMMENDED_ACTIONS.get(
        (category, priority),
        "Review and route to the appropriate team for follow-up.",
    )
    

def get_suggested_response(category: str, priority: str, sentiment_label: str) -> dict:
    # Returns {'subject': str, 'body': str}. 
    # Marked clearly as a SUGGESTED response — the system never sends automatically
    category_templates = _RESPONSE_TEMPLATES.get(category, _RESPONSE_TEMPLATES["General Inquiry"])
    body = category_templates.get(sentiment_label, category_templates["Neutral"])
    subject = _SUBJECT_PREFIX.get(category, "Re: Your Message")
    return {"subject": subject, "body": body}
