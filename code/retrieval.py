# code/retrieval.py

import os
import re
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from rank_bm25 import BM25Okapi


SUPPORTED_COMPANIES = {"hackerrank", "claude", "visa"}

MIN_CHUNK_SIZE = 80
MAX_CHUNK_SIZE = 900
TOP_K = 5
MAX_CONTEXT_CHARS = 4000


def normalize_text(text: Optional[str]) -> str:
    if text is None:
        return ""

    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def tokenize(text: Optional[str]) -> List[str]:
    normalized = normalize_text(text)

    if not normalized:
        return []

    seen = set()
    tokens = []

    for token in normalized.split():
        if token not in seen:
            seen.add(token)
            tokens.append(token)

    return tokens


def generate_chunk_id(source_path: str, chunk_text: str) -> str:
    base = f"{source_path}:{chunk_text[:300].strip()}"
    return uuid.uuid5(uuid.NAMESPACE_URL, base).hex


def infer_company_from_path(path: str) -> Optional[str]:
    lowered = path.lower()

    for company in SUPPORTED_COMPANIES:
        if company in lowered:
            return company

    return None


def infer_product_area(path: str, title: str, content: str) -> str:
    combined = f"{path} {title} {content}".lower()

    product_map = {
        "billing": ["billing", "invoice", "payment", "subscription"],
        "authentication": ["login", "authentication", "password", "2fa"],
        "events": ["event", "webinar", "registration"],
        "assessments": ["assessment", "challenge", "test"],
        "cards": ["card", "visa card", "debit", "credit"],
        "payments": ["payment", "transaction", "refund"],
        "account_access": ["account", "access", "locked"],
        "onboarding": ["onboarding", "setup", "getting started"],
    }

    for area, keywords in product_map.items():
        if any(keyword in combined for keyword in keywords):
            return area

    return "general"


def safe_read_markdown(file_path: Path) -> str:
    encodings = ["utf-8", "utf-8-sig", "latin-1"]

    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding, errors="ignore") as file:
                return file.read()
        except Exception:
            continue

    return ""


def extract_title(content: str, fallback: str) -> str:
    for line in content.splitlines():
        stripped = line.strip()

        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()

    return fallback


def split_markdown_chunks(content: str) -> List[str]:
    content = content.replace("\r\n", "\n")

    sections = re.split(r"\n(?=#)", content)

    chunks = []

    for section in sections:
        section = section.strip()

        if not section:
            continue

        paragraphs = re.split(r"\n\s*\n", section)

        current_chunk = ""

        for paragraph in paragraphs:
            paragraph = paragraph.strip()

            if not paragraph:
                continue

            candidate = f"{current_chunk}\n\n{paragraph}".strip()

            if len(candidate) <= MAX_CHUNK_SIZE:
                current_chunk = candidate
            else:
                if len(current_chunk) >= MIN_CHUNK_SIZE:
                    chunks.append(current_chunk)

                current_chunk = paragraph

        if len(current_chunk) >= MIN_CHUNK_SIZE:
            chunks.append(current_chunk)

    return chunks


def load_corpus(data_dir: str = "../data") -> List[Dict]:
    documents = []
    seen_chunk_ids = set()

    data_path = Path(data_dir)

    if not data_path.exists():
        print("[WARN] Data directory missing")
        return []

    markdown_files = list(data_path.rglob("*.md"))

    for file_path in markdown_files:
        try:
            raw_content = safe_read_markdown(file_path)

            if not raw_content.strip():
                continue

            title = extract_title(raw_content, file_path.stem)

            company = infer_company_from_path(str(file_path))

            if company not in SUPPORTED_COMPANIES:
                continue

            chunks = split_markdown_chunks(raw_content)

            for chunk in chunks:
                cleaned_chunk = chunk.strip()

                if len(cleaned_chunk) < MIN_CHUNK_SIZE:
                    continue

                chunk_id = generate_chunk_id(str(file_path), cleaned_chunk)

                if chunk_id in seen_chunk_ids:
                    continue

                seen_chunk_ids.add(chunk_id)

                document = {
                    "chunk_id": chunk_id,
                    "company": company,
                    "source_path": str(file_path),
                    "title": title,
                    "product_area": infer_product_area(
                        str(file_path),
                        title,
                        cleaned_chunk,
                    ),
                    "content": cleaned_chunk,
                }

                documents.append(document)

        except Exception as error:
            print(f"[WARN] Failed loading {file_path}: {error}")

    return documents


class RetrievalEngine:
    def __init__(self, documents: List[Dict]):
        self.documents = documents or []

        self.tokenized_corpus = []

        for document in self.documents:
            tokens = tokenize(
                f"{document.get('title', '')} {document.get('content', '')}"
            )

            self.tokenized_corpus.append(tokens if tokens else ["empty"])

        self.bm25 = (
            BM25Okapi(self.tokenized_corpus)
            if self.tokenized_corpus
            else None
        )

    def search(
        self,
        issue: Optional[str],
        subject: Optional[str] = None,
        company: Optional[str] = None,
        top_k: int = TOP_K,
    ) -> Dict:
        if not self.documents or self.bm25 is None:
            return {
                "results": [],
                "confidence": "low",
            }

        query = normalize_text(f"{subject or ''} {issue or ''}")

        if not query:
            return {
                "results": [],
                "confidence": "low",
            }

        query_tokens = tokenize(query)

        if not query_tokens:
            return {
                "results": [],
                "confidence": "low",
            }

        candidate_docs = self.documents

        if company in SUPPORTED_COMPANIES:
            candidate_docs = [
                doc
                for doc in self.documents
                if doc["company"] == company
            ]

        if not candidate_docs:
            return {
                "results": [],
                "confidence": "low",
            }

        candidate_indices = [
            self.documents.index(doc)
            for doc in candidate_docs
        ]

        bm25_scores = self.bm25.get_scores(query_tokens)

        scored_results = []

        for index in candidate_indices:
            document = self.documents[index]
            score = float(bm25_scores[index])

            title = normalize_text(document.get("title", ""))

            title_tokens = tokenize(title)

            overlap = len(set(query_tokens) & set(title_tokens))

            score += overlap * 2.0

            if company and document["company"] == company:
                score += 1.5

            if query in normalize_text(document["content"]):
                score += 3.0

            scored_results.append((score, document))

        scored_results.sort(key=lambda item: item[0], reverse=True)

        selected = []
        seen_chunks = set()
        total_chars = 0

        for score, document in scored_results[: top_k * 2]:
            content = document["content"]

            normalized = normalize_text(content)

            if normalized in seen_chunks:
                continue

            if total_chars + len(content) > MAX_CONTEXT_CHARS:
                break

            seen_chunks.add(normalized)

            total_chars += len(content)

            selected.append(
                {
                    "score": round(score, 4),
                    "chunk_id": document["chunk_id"],
                    "company": document["company"],
                    "product_area": document["product_area"],
                    "source_path": document["source_path"],
                    "title": document["title"],
                    "content": document["content"],
                }
            )

            if len(selected) >= top_k:
                break

        confidence = "low"

        if selected:
            top_score = selected[0]["score"]

            if top_score >= 10:
                confidence = "high"
            elif top_score >= 4:
                confidence = "medium"

        return {
            "results": selected,
            "confidence": confidence,
        }