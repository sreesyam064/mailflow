def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json() == {"status": "healthy"}
    

def test_analyze_valid_email(client):
    resp = client.post(
        "/analyze",
        json={
            "subject": "Payment deducted but subscription inactive",
            "body": "I was charged $999 yesterday but my subscription is still inactive.",
        },
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert "category" in data and "label" in data["category"] and "confidence" in data["category"]
    assert "priority" in data and "label" in data["priority"]
    assert data["priority"]["label"] in ("LOW", "MEDIUM", "HIGH")
    assert "sentiment" in data
    assert "department" in data
    assert "entities" in data
    assert "recommended_action" in data
    assert "suggested_response" in data
    assert "requires_human_review" in data
    
    
def test_analyze_missing_subject(client):
    resp = client.post("/analyze", json={"body": "some body text"})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_analyze_missing_body(client):
    resp = client.post("/analyze", json={"subject": "some subject"})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_analyze_missing_both_fields(client):
    resp = client.post("/analyze", json={})
    assert resp.status_code == 400


def test_analyze_empty_input(client):
    resp = client.post("/analyze", json={"subject": "", "body": ""})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_analyze_malformed_json(client):
    resp = client.post(
        "/analyze",
        data="{not valid json",
        content_type="application/json",
    )
    assert resp.status_code == 400


def test_analyze_wrong_field_types(client):
    resp = client.post("/analyze", json={"subject": 123, "body": ["not", "a", "string"]})
    assert resp.status_code == 400


def test_analyze_excessively_large_input(client):
    huge_body = "x" * 30000
    resp = client.post("/analyze", json={"subject": "x", "body": huge_body})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_analyze_response_structure_types(client):
    resp = client.post(
        "/analyze",
        json={"subject": "Server down", "body": "Our production server is down."},
    )
    data = resp.get_json()
    assert isinstance(data["category"]["confidence"], float)
    assert isinstance(data["requires_human_review"], bool)
    assert isinstance(data["entities"], dict)
    assert isinstance(data["suggested_response"], dict)
    assert "subject" in data["suggested_response"] and "body" in data["suggested_response"]
    assert "review_reason" in data
    assert isinstance(data["review_reason"], dict)


def test_analyze_low_information_input_triggers_review(client):
    # Body under the low-information threshold should force human review
    # even if the underlying model happened to be confident.
    resp = client.post(
        "/analyze",
        json={"subject": "Help", "body": "please help"},
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["review_reason"]["low_information_input"] is True
    assert data["requires_human_review"] is True


def test_analyze_missing_subject_triggers_review(client):
    resp = client.post(
        "/analyze",
        json={"subject": "", "body": "This is a reasonably detailed email body with real content in it."},
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["review_reason"]["low_information_input"] is True
