# code/main.py

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from llm import generate_response
from retrieval import RetrievalEngine, load_corpus
from triage import (
    build_justification,
    classify_request_type,
    generate_fallback_response,
    infer_company,
    infer_product_area,
    should_escalate,
)


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

INPUT_CSV = PROJECT_ROOT / "support_tickets" / "support_tickets.csv"
OUTPUT_CSV = PROJECT_ROOT / "support_tickets" / "output.csv"
DATA_DIR = PROJECT_ROOT / "data"


REQUIRED_COLUMNS = [
    "status",
    "product_area",
    "response",
    "justification",
    "request_type",
]


def safe_string(value):
    if value is None:
        return ""

    if isinstance(value, float) and np.isnan(value):
        return ""

    return str(value).strip()


def validate_input_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required_input_columns = {"issue", "subject", "company"}

    missing = required_input_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required input columns: {sorted(missing)}"
        )

    return df.fillna("")


def process_ticket(row, retrieval_engine: RetrievalEngine):
    issue = safe_string(row.get("issue"))
    subject = safe_string(row.get("subject"))
    provided_company = safe_string(row.get("company")).lower()

    inferred_company = infer_company(issue, subject)

    company = (
        provided_company
        if provided_company in {"hackerrank", "claude", "visa"}
        else inferred_company
    )

    retrieval_results = retrieval_engine.search(
        issue=issue,
        subject=subject,
        company=company,
    )

    request_type = classify_request_type(issue, subject)

    product_area = infer_product_area(
        issue=issue,
        subject=subject,
        retrieval_results=retrieval_results,
    )

    escalate = should_escalate(
        issue=issue,
        subject=subject,
        company=company,
        retrieval_results=retrieval_results,
    )

    status = "escalated" if escalate else "replied"

    if status == "replied":
        response = generate_response(
            issue=issue,
            subject=subject,
            retrieval_results=retrieval_results,
            status=status,
        )
    else:
        response = generate_fallback_response(status)

    justification = build_justification(
        status=status,
        retrieval_results=retrieval_results,
        company=company,
    )

    return {
        "status": safe_string(status),
        "product_area": safe_string(product_area),
        "response": safe_string(response),
        "justification": safe_string(justification),
        "request_type": safe_string(request_type),
    }


def main():
    print("=" * 70)
    print("SUPPORT TRIAGE ENGINE")
    print("=" * 70)

    if not INPUT_CSV.exists():
        print(f"[ERROR] Missing input CSV: {INPUT_CSV}")
        sys.exit(1)

    print("[INFO] Loading markdown corpus...")

    documents = load_corpus(str(DATA_DIR))

    print(f"[INFO] Loaded {len(documents)} retrieval chunks")

    retrieval_engine = RetrievalEngine(documents)

    print("[INFO] Loading support tickets...")

    try:
        df = pd.read_csv(INPUT_CSV)
        df = validate_input_dataframe(df)

    except Exception as error:
        print(f"[ERROR] Failed reading input CSV: {error}")
        sys.exit(1)

    output_rows = []

    total_rows = len(df)

    for index, row in df.iterrows():
        print(f"[INFO] Processing ticket {index + 1}/{total_rows}")

        try:
            processed = process_ticket(row, retrieval_engine)

        except Exception as error:
            processed = {
                "status": "escalated",
                "product_area": "general",
                "response": (
                    "The request could not be processed safely and "
                    "has been escalated for review."
                ),
                "justification": (
                    f"Pipeline fallback triggered due to processing "
                    f"error: {error}"
                ),
                "request_type": "invalid",
            }

        output_rows.append(processed)

    output_df = pd.DataFrame(output_rows)

    for column in REQUIRED_COLUMNS:
        if column not in output_df.columns:
            output_df[column] = ""

    output_df = output_df[REQUIRED_COLUMNS]

    output_df = output_df.fillna("")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    try:
        output_df.to_csv(
            OUTPUT_CSV,
            index=False,
            encoding="utf-8",
        )

        print(f"[SUCCESS] Output written to: {OUTPUT_CSV}")

    except Exception as error:
        print(f"[ERROR] Failed writing output CSV: {error}")
        sys.exit(1)

    print("=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()