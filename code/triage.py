import re
from typing import Dict, List


ESCALATION_KEYWORDS = [
    "fraud",
    "hacked",
    "stolen card",
    "unauthorized payment",
    "unauthorised payment",
    "legal",
    "lawsuit",
    "billing dispute",
    "account takeover",
    "refund escalation",
    "human agent",
    "security breach",
    "chargeback",
    "scam",
    "identity theft",
]


REQUEST_TYPE_RULES = {
    "feature_request": [
        "feature request",
        "would like",
        "please add",
        "new feature",
        "enhancement",
        "improve",
    ],
    "bug": [
        "bug",
        "error",
        "broken",
        "fails",
        "failure",
        "issue",
        "not working",
        "crash",
    ],
    "product_issue": [
        "cannot",
        "unable",
        "problem",
        "question",
        "support",
        "help",
    ],
}


def normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def classify_request_type(text: str) -> str:
    text = normalize(text)

    for request_type, keywords in REQUEST_TYPE_RULES.items():
        for keyword in keywords:
            if keyword in text:
                return request_type

    if len(text) < 5:
        return "invalid"

    return "product_issue"


def should_escalate(
    ticket_text: str,
    company: str,
    retrieval_results: List[Dict],
) -> Dict:
    text = normalize(ticket_text)

    for keyword in ESCALATION_KEYWORDS:
        if keyword in text:
            return {
                "escalate": True,
                "reason": f"Sensitive keyword detected: {keyword}",
            }

    if not company:
        return {
            "escalate": True,
            "reason": "Unable to confidently infer company",
        }

    if not retrieval_results:
        return {
            "escalate": True,
            "reason": "No relevant documents retrieved",
        }

    top_score = retrieval_results[0].get("score", 0)

    if top_score < 1.0:
        return {
            "escalate": True,
            "reason": "Retrieval confidence too weak",
        }

    return {
        "escalate": False,
        "reason": "Safe automated reply possible",
    }


def infer_product_area(retrieval_results: List[Dict]) -> str:
    if not retrieval_results:
        return "general"

    top_doc = retrieval_results[0]

    source = top_doc.get("source", "").lower()

    if "billing" in source:
        return "billing"

    if "auth" in source or "login" in source:
        return "authentication"

    if "payment" in source:
        return "payments"

    if "api" in source:
        return "api"

    company = top_doc.get("company", "").strip()

    if company:
        return company

    return "general"