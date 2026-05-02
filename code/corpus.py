import re
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from chunking import split_markdown_chunks


SUPPORTED_COMPANIES = {"hackerrank", "claude", "visa"}


PRODUCT_AREA_RULES = {
    "billing": {
        "billing": 5,
        "invoice": 4,
        "subscription": 4,
        "refund": 3,
    },
    "authentication": {
        "login": 5,
        "authentication": 5,
        "password": 4,
        "otp": 3,
        "2fa": 4,
    },
    "events": {
        "event": 4,
        "webinar": 4,
        "registration": 3,
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
        "checkout": 3,
    },
    "cards": {
        "card": 5,
        "visa card": 6,
        "credit card": 5,
        "debit card": 5,
    },
    "account_access": {
        "account": 4,
        "access": 4,
        "locked": 5,
        "disabled": 4,
    },
    "api": {
        "api": 5,
        "sdk": 4,
        "endpoint": 4,
        "token": 3,
    },
    "onboarding": {
        "setup": 4,
        "onboarding": 5,
        "getting started": 5,
    },
}


def normalize_text(text: Optional[str]) -> str:
    if text is None:
        return ""

    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def generate_chunk_id(source_path: str, chunk_text: str) -> str:
    base = f"{source_path}:{chunk_text[:300].strip()}"
    return uuid.uuid5(uuid.NAMESPACE_URL, base).hex


def safe_read_markdown(file_path: Path) -> str:
    encodings = ["utf-8", "utf-8-sig", "latin-1"]

    for encoding in encodings:
        try:
            with open(
                file_path,
                "r",
                encoding=encoding,
                errors="ignore",
            ) as file:
                return file.read()
        except Exception:
            continue

    return ""


def infer_company_from_path(path: str) -> Optional[str]:
    lowered = path.lower()

    for company in SUPPORTED_COMPANIES:
        if company in lowered:
            return company

    return None


def extract_title(content: str, fallback: str) -> str:
    for line in content.splitlines():
        stripped = line.strip()

        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()

            if title:
                return title[:200]

    return fallback[:200]


def infer_product_area(
    source_path: str,
    title: str,
    content: str,
) -> str:
    combined = normalize_text(
        f"{source_path} {title} {content}"
    )

    scores = {}

    for area, keywords in PRODUCT_AREA_RULES.items():
        area_score = 0

        for keyword, weight in keywords.items():
            if keyword in combined:
                area_score += weight

        if area_score > 0:
            scores[area] = area_score

    if not scores:
        return "general"

    return max(scores.items(), key=lambda x: x[1])[0]


def build_document(
    company: str,
    source_path: str,
    title: str,
    product_area: str,
    content: str,
) -> Dict:
    return {
        "chunk_id": generate_chunk_id(source_path, content),
        "company": company or "",
        "source_path": source_path or "",
        "title": title or "",
        "product_area": product_area or "general",
        "content": content or "",
    }


def load_corpus(data_dir: str) -> List[Dict]:
    data_path = Path(data_dir)

    if not data_path.exists():
        print("[WARN] Data directory missing")
        return []

    markdown_files = list(data_path.rglob("*.md"))

    documents = []

    seen_chunk_ids = set()
    seen_documents = set()

    for file_path in markdown_files:
        try:
            raw_content = safe_read_markdown(file_path)

            if not raw_content.strip():
                continue

            source_path = str(file_path)

            if source_path in seen_documents:
                continue

            seen_documents.add(source_path)

            company = infer_company_from_path(source_path)

            if company not in SUPPORTED_COMPANIES:
                continue

            title = extract_title(
                raw_content,
                file_path.stem,
            )

            chunks = split_markdown_chunks(raw_content)

            for chunk in chunks:
                cleaned_chunk = chunk.strip()

                if not cleaned_chunk:
                    continue

                product_area = infer_product_area(
                    source_path,
                    title,
                    cleaned_chunk,
                )

                document = build_document(
                    company=company,
                    source_path=source_path,
                    title=title,
                    product_area=product_area,
                    content=cleaned_chunk,
                )

                chunk_id = document["chunk_id"]

                if chunk_id in seen_chunk_ids:
                    continue

                seen_chunk_ids.add(chunk_id)

                documents.append(document)

        except Exception as error:
            print(f"[WARN] Failed loading {file_path}: {error}")

    return documents