import os
import sys
import traceback

import numpy as np
import pandas as pd

from dotenv import load_dotenv

from retrieval import RetrievalEngine
from triage import (
    classify_request_type,
    infer_product_area,
    should_escalate,
)
from llm import GeminiResponder


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")

INPUT_CSV = os.path.join(
    BASE_DIR,
    "support_tickets",
    "support_tickets.csv",
)

OUTPUT_CSV = os.path.join(
    BASE_DIR,
    "support_tickets",
    "output.csv",
)


def safe_string(value):
    if value is None:
        return ""

    if isinstance(value, float) and np.isnan(value):
        return ""

    return str(value).strip()


def load_tickets():
    if not os.path.exists(INPUT_CSV):
        print(f"[ERROR] Missing CSV: {INPUT_CSV}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(INPUT_CSV)

        df = df.fillna("")

        return df

    except Exception as e:
        print(f"[ERROR] Failed to load CSV: {e}")
        return pd.DataFrame()


def build_ticket_text(row):
    parts = []

    for value in row.values:
        text = safe_string(value)

        if text:
            parts.append(text)

    return " ".join(parts).strip()


def process_ticket(ticket_text, retrieval_engine, llm):
    request_type = classify_request_type(ticket_text)

    company = retrieval_engine.infer_company(ticket_text)

    retrieval_results = retrieval_engine.retrieve(
        query=ticket_text,
        company=company,
        top_k=5,
    )

    escalation = should_escalate(
        ticket_text=ticket_text,
        company=company,
        retrieval_results=retrieval_results,
    )

    status = "escalated" if escalation["escalate"] else "replied"

    product_area = infer_product_area(retrieval_results)

    if status == "escalated":
        response = (
            "Your request has been forwarded to a human support specialist for further review."
        )
    else:
        response = llm.generate_response(
            ticket=ticket_text,
            context_chunks=retrieval_results,
        )

    response = safe_string(response)

    if not response:
        response = (
            "Your request has been forwarded to a human support specialist for further review."
        )

    return {
        "status": status,
        "product_area": product_area,
        "response": response,
        "justification": escalation["reason"],
        "request_type": request_type,
    }


def ensure_output_directory():
    output_dir = os.path.dirname(OUTPUT_CSV)

    os.makedirs(output_dir, exist_ok=True)


def main():
    print("[INFO] Starting Phase 1 triage engine")

    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    retrieval_engine = RetrievalEngine(DATA_DIR)

    print("[INFO] Loading markdown corpus")

    retrieval_engine.load_documents()

    if not retrieval_engine.documents:
        print("[WARN] No markdown documents loaded")

    print(
        f"[INFO] Loaded {len(retrieval_engine.documents)} document chunks"
    )

    llm = GeminiResponder(api_key)

    tickets_df = load_tickets()

    if tickets_df.empty:
        print("[WARN] No tickets found")

    results = []

    for idx, row in tickets_df.iterrows():
        try:
            ticket_text = build_ticket_text(row)

            if not ticket_text:
                result = {
                    "status": "escalated",
                    "product_area": "general",
                    "response": (
                        "Your request has been forwarded to a human support specialist for further review."
                    ),
                    "justification": "Empty ticket content",
                    "request_type": "invalid",
                }

            else:
                result = process_ticket(
                    ticket_text=ticket_text,
                    retrieval_engine=retrieval_engine,
                    llm=llm,
                )

            results.append(result)

            print(
                f"[INFO] Processed ticket {idx + 1}/{len(tickets_df)}"
            )

        except Exception as e:
            print(f"[ERROR] Ticket processing failed: {e}")

            traceback.print_exc()

            results.append(
                {
                    "status": "escalated",
                    "product_area": "general",
                    "response": (
                        "Your request has been forwarded to a human support specialist for further review."
                    ),
                    "justification": "Unhandled processing failure",
                    "request_type": "invalid",
                }
            )

    output_df = pd.DataFrame(
        results,
        columns=[
            "status",
            "product_area",
            "response",
            "justification",
            "request_type",
        ],
    )

    output_df = output_df.fillna("")

    ensure_output_directory()

    try:
        output_df.to_csv(
            OUTPUT_CSV,
            index=False,
            encoding="utf-8",
        )

        print(f"[INFO] Output written to: {OUTPUT_CSV}")

    except Exception as e:
        print(f"[ERROR] Failed to write output CSV: {e}")
        sys.exit(1)

    print("[INFO] Phase 1 triage complete")


if __name__ == "__main__":
    main()