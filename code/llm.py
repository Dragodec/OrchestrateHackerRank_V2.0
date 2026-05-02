import os
import time
from typing import Dict, List

from dotenv import load_dotenv

try:
    from google import genai
except Exception:
    genai = None


load_dotenv()

MODEL_NAME = "gemini-2.5-flash"

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2


SYSTEM_PROMPT = """
You are a grounded support triage assistant.

STRICT RULES:
- Use ONLY provided context
- NEVER invent policies
- NEVER use outside knowledge
- NEVER hallucinate troubleshooting
- Keep responses concise
- If evidence is weak, say escalation is needed
"""


def build_context(results: List[Dict]) -> str:
    blocks = []

    seen = set()

    for result in results:
        title = result.get("title", "").strip()
        content = result.get("content", "").strip()

        normalized = content.lower()

        if (
            not content
            or normalized in seen
        ):
            continue

        seen.add(normalized)

        compact = (
            f"[{title}]\n{content}"
        )

        blocks.append(compact)

    return "\n\n".join(blocks)


def fallback_response() -> str:
    return (
        "We could not confidently generate a grounded "
        "response from the available support documentation."
    )


def generate_response(
    issue: str,
    subject: str,
    retrieval_results: Dict,
    status: str,
) -> str:
    if status == "escalated":
        return (
            "Your request requires additional review "
            "by a support specialist."
        )

    results = retrieval_results.get("results", [])

    if not results:
        return fallback_response()

    context = build_context(results)

    if not context.strip():
        return fallback_response()

    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key or genai is None:
        return fallback_response()

    try:
        client = genai.Client(api_key=api_key)
    except Exception:
        return fallback_response()

    prompt = f"""
ISSUE SUBJECT:
{subject}

ISSUE:
{issue}

SUPPORT CONTEXT:
{context}

TASK:
Generate a concise grounded support reply.
"""

    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=f"{SYSTEM_PROMPT}\n\n{prompt}",
            )

            text = getattr(response, "text", "")

            if text and text.strip():
                return text.strip()

        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY_SECONDS)

    return fallback_response()