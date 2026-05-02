# code/llm.py

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
- Use ONLY the provided support context.
- NEVER invent policies.
- NEVER invent troubleshooting steps.
- NEVER use outside knowledge.
- NEVER assume unsupported details.
- Keep responses concise and user-safe.
- If context is insufficient, say the issue requires escalation.
"""


def build_context(results: List[Dict]) -> str:
    if not results:
        return ""

    context_parts = []

    for result in results:
        title = result.get("title", "").strip()
        content = result.get("content", "").strip()

        if not content:
            continue

        block = f"TITLE: {title}\nCONTENT:\n{content}"

        context_parts.append(block)

    return "\n\n---\n\n".join(context_parts)


def fallback_response() -> str:
    return (
        "We could not confidently generate a grounded response from the "
        "available support documentation. Your request may require "
        "additional review."
    )


def generate_response(
    issue: str,
    subject: str,
    retrieval_results: Dict,
    status: str,
) -> str:
    if status == "escalated":
        return (
            "Your request requires additional review by a support "
            "specialist and has been escalated safely."
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

    user_prompt = f"""
SUPPORT ISSUE:
Subject: {subject or ''}

Issue:
{issue or ''}

GROUNDING CONTEXT:
{context}

TASK:
Generate a concise support response using ONLY the provided grounding context.
"""

    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=f"{SYSTEM_PROMPT}\n\n{user_prompt}",
            )

            text = ""

            if hasattr(response, "text") and response.text:
                text = response.text.strip()

            if text:
                return text

        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY_SECONDS)

    return fallback_response()