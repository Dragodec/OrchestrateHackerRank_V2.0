import re
from typing import Dict, Optional


SUPPORTED_COMPANIES = {"hackerrank", "claude", "visa"}

SENSITIVE_KEYWORDS = {
    "fraud",
    "hacked",
    "stolen card",
    "unauthorized payment",
    "account takeover",
    "lawsuit",
    "legal",
    "security breach",
    "refund dispute",
    "chargeback",
}


PRODUCT_AREA_KEYWORDS = {
    "billing": {
        "billing": 5,
        "invoice": 4,
        "refund": 4,
        "subscription": 4,
    },
    "authentication": {
        "login": 5,
        "password": 5,
        "otp": 4,
        "2fa": 4,
    },
    "assessments": {
        "assessment": 5,
        "challenge": 4,
        "coding test": 5,
    },
    "payments": {
        "payment": 5,
        "transaction": 4,
        "declined": 3,
    },
    "cards": {
        "card": 5,
        "credit card": 5,
        "debit card": 5,
    },
    "api": {
        "api": 5,
        "sdk": 4,
        "endpoint": 4,
    },
}


def normalize_text(text: Optional[str]) -> str:
    if text is None:
        return ""

    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def infer_company(
    issue: Optional[str],
    subject: Optional[str] = None,
):
    combined = normalize_text(
        f"{subject or ''} {issue or ''}"
    )

    company_scores = {
        "hackerrank": 0,
        "claude": 0,
        "visa": 0,
    }

    keyword_map = {
        "hackerrank": {
            "hackerrank": 5,
            "assessment": 3,
            "coding test": 4,
            "challenge": 3,
        },
        "claude": {
            "claude": 5,
            "anthropic": 5,
            "claude ai": 5,
        },
        "visa": {
            "visa": 5,
            "card": 2,
            "payment": 3,
            "transaction": 3,
        },
    }

    for company, keywords in keyword_map.items():
        for keyword, weight in keywords.items():
            if keyword in combined:
                company_scores[company] += weight

    ranked = sorted(
        company_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    top_company, top_score = ranked[0]

    if top_score <= 0:
        return None

    if len(ranked) > 1:
        second_score = ranked[1][1]

        if abs(top_score - second_score) <= 1:
            return None

    return top_company


def classify_request_type(
    issue: Optional[str],
    subject: Optional[str] = None,
) -> str:
    combined = normalize_text(
        f"{subject or ''} {issue or ''}"
    )

    if not combined:
        return "invalid"

    if any(
        phrase in combined
        for phrase in {
            "feature request",
            "please add",
            "new feature",
        }
    ):
        return "feature_request"

    if any(
        phrase in combined
        for phrase in {
            "bug",
            "broken",
            "crash",
            "error",
            "fails",
        }
    ):
        return "bug"

    return "product_issue"


def infer_product_area(
    issue: Optional[str],
    subject: Optional[str] = None,
    retrieval_results: Optional[Dict] = None,
) -> str:
    combined = normalize_text(
        f"{subject or ''} {issue or ''}"
    )

    scores = {}

    for area, keywords in PRODUCT_AREA_KEYWORDS.items():
        score = 0

        for keyword, weight in keywords.items():
            if keyword in combined:
                score += weight

        if score > 0:
            scores[area] = score

    if retrieval_results:
        for result in retrieval_results.get(
            "results",
            [],
        ):
            area = result.get("product_area")

            if area:
                scores[area] = scores.get(area, 0) + 2

    if not scores:
        return "general"

    return max(
        scores.items(),
        key=lambda x: x[1],
    )[0]


def contains_sensitive_content(
    issue: Optional[str],
    subject: Optional[str] = None,
) -> bool:
    combined = normalize_text(
        f"{subject or ''} {issue or ''}"
    )

    return any(
        keyword in combined
        for keyword in SENSITIVE_KEYWORDS
    )


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

    confidence = retrieval_results.get(
        "confidence",
        "low",
    )

    if confidence == "low":
        return True

    if not retrieval_results.get("results"):
        return True

    return False


def build_justification(
    status: str,
    retrieval_results: Dict,
    company: Optional[str],
) -> str:
    confidence = retrieval_results.get(
        "confidence",
        "low",
    )

    if status == "escalated":
        if company not in SUPPORTED_COMPANIES:
            return (
                "Escalated due to unsupported or "
                "unclear company inference."
            )

        return (
            "Escalated due to insufficient "
            f"retrieval confidence ({confidence})."
        )

    top = retrieval_results["results"][0]

    return (
        "Replied using grounded retrieval evidence "
        f"from {top['source_path']} "
        f"with {confidence} confidence."
    )


def generate_fallback_response(status: str) -> str:
    return (
        "Your request requires additional review "
        "by a support specialist."
    )