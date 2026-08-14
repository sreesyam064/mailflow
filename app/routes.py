"""
Flask REST API

Two endpoints: GET /api/health, POST /api/analyze
Validation covers: missing subject/body, empty email, malformed JSON, 
excessively large input, and model/prediction errors — each returns a 
clear 4xx with a safe (non-leaky) error message.
"""

import logging

from flask import Blueprint, jsonify, request
from werkzeug.exceptions import BadRequest

from app.config import MAX_EMAIL_LENGTH, MIN_EMAIL_LENGTH
from app.entities import extract_entities
from app.features import extract_numeric_features
from app.predictor import predict_category, predict_priority
from app.responses import get_recommended_action, get_suggested_response
from app.router import route as route_department
from app.sentiment import analyze_sentiment

logger = logging.getLogger(__name__)
bp = Blueprint("api", __name__)

@bp.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200

@ bp.route("/analyze", methods=["POST"])
def analyze():
    # Malformed JSON
    try:
        payload = request.get_json(force=False, silent=False)
    except BadRequest:
        logger.warning("Received malformed JSON")
        return jsonify({"error": "Request body must be valid JSON."}), 400
    
    if payload is None or not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400
    
    subject  = payload.get("subject", "")
    body = payload.get("body", "")
    
    # Missing fields
    if "subject" not in payload or "body" not in payload:
        return jsonify({"error": "Both 'subject' and 'body' fields are required."}), 400
    
    if not isinstance(subject, str) or not isinstance(body, str):
        return jsonify({"error": "'subject' and 'body' must be strings."}), 400
    
    combined_length = len(subject) + len(body)
    
    # Empty input
    if combined_length < MIN_EMAIL_LENGTH:
        return jsonify({"error": "Email subject/body is empty or too short to analyze."}), 400
    
    # Excessively large input
    if combined_length > MAX_EMAIL_LENGTH:
        return jsonify({
            "error": f"Email exceeds maximum allowed length of {MAX_EMAIL_LENGTH} characters."
        }), 400
        
    try:
        text = f"{subject} {subject} {body}".strip()    # subject weighted, matches training prep
        
        sentiment = analyze_sentiment(text)
        category = predict_category(text)
        priority = predict_priority(text, sentiment["compound"])
        entities = extract_entities(body)
        department = route_department(category["label"])
        recommended_action = get_recommended_action(category["label"], priority["label"])
        suggested_response = get_suggested_response(category["label"], priority["label"], sentiment["label"])

        # Engineered signals used ONLY as human-review triggers, never as classifier input 
        signals = extract_numeric_features(subject, body)
        low_information_input = bool(
            signals["missing_subject"] or signals["low_information_body"]
        )
        
        requires_human_review = category["requires_human_review"] or priority["requires_human_review"] or low_information_input
        
        response = {
            "category": {
                "label": category["label"],
                "confidence": category["confidence"],
            },
            "priority": {
                "label": priority["label"],
                "confidence": priority["confidence"],
                "raw_model_label": priority["raw_model_label"],
                "sentiment_adjusted": priority["sentiment_adjusted"],
            },
            "sentiment": sentiment["label"],
            "department": department["department"],
            "department_explanation": department["explanation"],
            "entities": entities,
            "recommended_action": recommended_action,
            "suggested_response": suggested_response,
            "requires_human_review": requires_human_review,
            "review_reason": {
                "low_confidence_category": category["requires_human_review"],
                "low_confidence_priority": priority["requires_human_review"],
                "low_information_input": low_information_input,
            }
        }
        return jsonify(response), 200
    
    except Exception:
        logger.exception("Error during email analysis")
        return jsonify({"error": "An internal error occurred while analyzing the email."}), 500
    