# code/triage.py

import re
from typing import Dict, Optional


SUPPORTED_COMPANIES = {"hackerrank", "claude", "visa"}

SENSITIVE_KEYWORDS = {
    "fraud",
    "hacked",
    "stolen card",
    "unauthorized payment",
    "unauthorised payment",
    "account takeover",
    "lawsuit",
    "legal",
    "security breach",
    "refund dispute",
    "billing dispute",
    "chargeback",
    "human agent",
    "speak to human",
    "urgent escalation",
}

PRODUCT_AREA_KEYWORDS = {
    "billing": [
        "billing",
        "invoice",
        "payment",
        "refund",
        "charged",
        "subscription",
    ],
    "authentication": [
        "login",
        "password",
        "authentication",
        "2fa",
        "otp",
        "verification",
    ],
    "events": [
        "event",
        "registration",
        "webinar",
        "leaderboard",
    ],
    "assessments": [
        "assessment",
        "challenge",
        "coding test",
        "submission",
    ],
    "cards": [
        "card",
        "visa card",
        "credit card",
        "debit card",
    ],
    "payments": [
        "transaction",
        "payment",
        "checkout",
        "declined",
    ],
    "account_access": [
        "account",
        "locked",
        "access",
        "disabled",
    ],
    "onboarding": [
        "setup",
        "getting started",
        "onboarding",
    ],
}


def normalize_text(text: Optional[str]) -> str:
    if text is None:
        return ""

    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def infer_company(issue: Optional[str], subject: Optional[str] = None):
    combined = normalize_text(f"{subject or ''} {issue or ''}")

    company_keywords = {
        "hackerrank": [
            "hackerrank",
            "coding challenge",
            "assessment",
            "hrx",
        ],
        "claude": [
            "claude",
            "anthropic",
            "claude ai",
        ],
        "visa": [
            "visa",
            "visa card",
            "credit card",
            "debit card",
            "payment declined",
        ],
    }

    matched = []

    for company, keywords in company_keywords.items():
        if any(keyword in combined for keyword in keywords):
            matched.append(company)

    if len(matched) == 1:
        return matched[0]

    return None


def classify_request_type(
    issue: Optional[str],
    subject: Optional[str] = None,
) -> str:
    combined = normalize_text(f"{subject or ''} {issue or ''}")

    if not combined:
        return "invalid"

    feature_keywords = {
        "feature request",
        "please add",
        "would like",
        "new feature",
        "feature suggestion",
    }

    bug_keywords = {
        "bug",
        "broken",
        "not working",
        "error",
        "crash",
        "fails",
        "failure",
    }

    product_keywords = {
        "how do i",
        "unable",
        "issue",
        "problem",
        "support",
        "help",
        "question",
    }

    if any(keyword in combined for keyword in feature_keywords):
        return "feature_request"

    if any(keyword in combined for keyword in bug_keywords):
        return "bug"

    if any(keyword in combined for keyword in product_keywords):
        return "product_issue"

    return "invalid"


def infer_product_area(
    issue: Optional[str],
    subject: Optional[str] = None,
    retrieval_results: Optional[Dict] = None,
) -> str:
    combined = normalize_text(f"{subject or ''} {issue or ''}")

    for area, keywords in PRODUCT_AREA_KEYWORDS.items():
        if any(keyword in combined for keyword in keywords):
            return area

    if retrieval_results:
        results = retrieval_results.get("results", [])

        if results:
            top_result = results[0]

            retrieval_area = top_result.get("product_area")

            if retrieval_area:
                return retrieval_area

    return "general"


def contains_sensitive_content(
    issue: Optional[str],
    subject: Optional[str] = None,
) -> bool:
    combined = normalize_text(f"{subject or ''} {issue or ''}")

    return any(keyword in combined for keyword in SENSITIVE_KEYWORDS)


def should_escalate(
    issue: Optional[str],
    subject: Optional[str],
    company: Optional[str],
    retrieval_results: Dict,
) -> bool:
    if contains_sensitive_content(issue, subject):
        return True

    if company not in SUPPORTED_COMPANIES:
        return True

    confidence = retrieval_results.get("confidence", "low")

    if confidence == "low":
        return True

    results = retrieval_results.get("results", [])

    if not results:
        return True

    return False


def build_justification(
    status: str,
    retrieval_results: Dict,
    company: Optional[str],
) -> str:
    if status == "escalated":
        confidence = retrieval_results.get("confidence", "low")

        if company not in SUPPORTED_COMPANIES:
            return (
                "Escalated because company could not be "
                "confidently inferred."
            )

        if confidence == "low":
            return (
                "Escalated due to weak retrieval confidence "
                "or insufficient support evidence."
            )

        return "Escalated due to sensitive or high-risk content."

    results = retrieval_results.get("results", [])

    if not results:
        return "Replied using limited grounded support documentation."

    source = results[0].get("source_path", "support corpus")

    return f"Replied using grounded retrieval evidence from {source}."


def generate_fallback_response(status: str) -> str:
    if status == "escalated":
        return (
            "Your request requires additional review by a support "
            "specialist. The issue has been escalated for further assistance."
        )

    return (
        "Based on the available support documentation, we found "
        "guidance related to your request."
    )