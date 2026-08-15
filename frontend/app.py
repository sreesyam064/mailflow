"""
MailFlow AI frontend

Calls Flask REST API over HTTP (never by importing Flask directly) to keep
frontend/backend seperation clean. In single-container deployment (Docker on 
Render pr other platforms), Gunicorn runs the Flask API on 127.0.0.1:5000 as its
own process — started by UI-only: it never starts, owns, or manages Flask process.
"""

import os

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_URL", "http://127.0.0.1:5000")

st.set_page_config(page_title="MailFlow AI", page_icon="📧", layout="wide")

st.title("📧 MailFlow AI")
st.caption("Intelligent Email Classification & Routing — a human-in-the-loop triage assistant")

SAMPLE_EMAILS = {
    "— Select a sample email —": ("", ""),
    "Billing": (
        "Payment deducted but subscription inactive",
        "I was charged $999 yesterday but my subscription is still inactive. Please help resolve this quickly.",
    ),
    "Technical Support": (
        "Server down, urgent help needed",
        "Our production server is down and throwing 500 errors on every request. This started after the latest deployment. We need urgent technical assistance to restore service.",
    ),
    "Sales": (
        "Request for sales quote",
        "Could you provide a sales quote for the enterprise plan? We are evaluating vendors and would like pricing details and a demo.",
    ),
    "General Inquiry": (
        "General question about your services",
        "Hi, I had a general question about your services and wanted to know more before deciding whether to sign up.",
    ),
    "General Inquiry (positive feedback)": (
        "Dashboard redesign feedback",
        "The new dashboard is much easier to use. Great work on the redesign, the navigation is far more intuitive now.",
    ),
}

with st.sidebar:
    st.subheader("Try a sample email")
    choice = st.selectbox("Sample emails", list(SAMPLE_EMAILS.keys()), label_visibility="collapsed")
    if choice != "— Select a sample email —":
        st.session_state["subject_input"] = SAMPLE_EMAILS[choice][0]
        st.session_state["body_input"] = SAMPLE_EMAILS[choice][1]
        
    st.divider()
    st.caption(
        "MailFlow AI assists human operators — it never sends emails "
        "automatically. All suggested responses require human review before sending."
    )
    
col_input, col_output = st.columns([1, 1.3], gap="large")

with col_input:
    st.subheader("Incoming Email")
    subject = st.text_input("Subject", key="subject_input", placeholder="e.g. Payment deducted but subscription inactive")
    body = st.text_area("Email body", key="body_input", height=220, placeholder="Paste or type the email body here...")
    analyze_clicked = st.button("Analyze Email", type="primary", use_container_width=True)
    
with col_output:
    st.subheader("Analysis")
    
    if analyze_clicked:
        if not subject.strip() and not body.strip():
            st.warning("Please enter a subject or body before analyzing.")
        else:
            try:
                with st.spinner("Analyzing email..."):
                    resp = requests.post(
                        f"{API_BASE_URL}/analyze",
                        json={"subject": subject, "body": body},
                        timeout=10,
                    )
                if resp.status_code != 200:
                    st.error(resp.json().get("error", "Analysis failed."))
                else:
                    result = resp.json()
                    st.session_state["last_result"] = result
            except requests.exceptions.RequestException as e:
                st.error(f"Could not reach the MailFlow API: {e}")
                
    result = st.session_state.get("last_result")
    
    if result:
        if result.get("requires_human_review"):
            st.warning("⚠️ Human review recommended — model confidence is below threshold.")
            
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Intent", result["category"]["label"], f"{result['category']['confidence']*100:.0f}% conf.")
        
        priority_label = result["priority"]["label"]
        priority_delta = f"{result['priority']['confidence']*100:.0f}% conf."
        if result["priority"]["sentiment_adjusted"]:
            priority_delta += " · sentiment-adjusted"
        m2.metric("Priority", priority_label, priority_delta)
        
        m3.metric("Sentiment", result["sentiment"])
        m4.metric("Routed to", result["department"])
        
        # Visualurgency cue — a triage dashboard needs HIGH to be instantly distinguishable from
        # LOW/MEDIUM at a glance, not just redeable in a metric label.
        priority_banner = {
            "HIGH":("🔴", "HIGH priority — recommend prioritizing this email."),
            "MEDIUM": ("🟠", "MEDIUM priority — handle within standard SLA."),
            "LOW": ("🟢", "LOW priority — standard queue."),
        }
        icon, banner_text = priority_banner.get(priority_label, ("⚪", "Priority unknown."))
        if priority_label == "HIGH":
            st.error(f"{icon} {banner_text}")
        elif priority_label == "MEDIUM":
            st.warning(f"{icon} {banner_text}")
        else:
            st.success(f"{icon} {banner_text}")
            
        if result["priority"]["sentiment_adjusted"]:
            st.caption(
                f"ℹ️ Priority was raised from the model's raw prediction "
                f"({result['priority']['raw_model_label']}) due to strongly negative sentiment. "
                f"This adjustment is applied transparently, never silently."
            )
            
        st.caption(result["department_explanation"])    
        
        st.markdown("#### Extracted Entities")
        entities = result["entities"]
        ent_cols = st.columns(len(entities))
        for col, (key, val) in zip(ent_cols, entities.items()):
            col.metric(key.replace("_", " ").title(), val if val else "—")
            
        st.markdown("#### Recommended Action")
        st.info(result["recommended_action"])
        
        st.markdown("#### Suggested Response  \n*(editable — review before sending)*")
        st.text_input("Subject line", value=result["suggested_response"]["subject"], key="resp_subject")
        st.text_area("Response body", value=result["suggested_response"]["body"], height=150, key="resp_body")
        
    else:
        st.caption("Enter an email and click **Analyze Email** to see results here.")
        