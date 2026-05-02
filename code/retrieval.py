import os
import re
from typing import List, Dict

from rank_bm25 import BM25Okapi


def safe_read_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    text = normalize_text(text)

    if not text:
        return []

    return re.findall(r"\b\w+\b", text)


def chunk_markdown(content: str) -> List[str]:
    if not content:
        return []

    chunks = re.split(r"\n\s*\n", content)

    cleaned = []

    for chunk in chunks:
        chunk = chunk.strip()

        if len(chunk) < 20:
            continue

        cleaned.append(chunk)

    return cleaned


class RetrievalEngine:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.documents = []
        self.tokenized_docs = []
        self.bm25 = None

    def load_documents(self):
        seen = set()
        doc_id = 0

        if not os.path.exists(self.data_dir):
            print(f"[WARN] Data directory missing: {self.data_dir}")
            return

        for root, _, files in os.walk(self.data_dir):
            for file_name in files:
                if not file_name.lower().endswith(".md"):
                    continue

                path = os.path.join(root, file_name)

                content = safe_read_file(path)

                if not content.strip():
                    continue

                company = os.path.basename(root).strip().lower()

                chunks = chunk_markdown(content)

                for chunk in chunks:
                    normalized = normalize_text(chunk)

                    if not normalized:
                        continue

                    dedupe_key = f"{path}:{normalized}"

                    if dedupe_key in seen:
                        continue

                    seen.add(dedupe_key)

                    tokens = tokenize(chunk)

                    if not tokens:
                        continue

                    self.documents.append(
                        {
                            "id": doc_id,
                            "company": company,
                            "source": path,
                            "text": chunk,
                        }
                    )

                    self.tokenized_docs.append(tokens)

                    doc_id += 1

        if self.tokenized_docs:
            self.bm25 = BM25Okapi(self.tokenized_docs)

    def infer_company(self, ticket_text: str) -> str:
        text = normalize_text(ticket_text)

        possible_companies = set()

        for doc in self.documents:
            company = doc.get("company", "").strip()

            if company and company in text:
                possible_companies.add(company)

        if len(possible_companies) == 1:
            return list(possible_companies)[0]

        return ""

    def retrieve(self, query: str, company: str, top_k: int = 5) -> List[Dict]:
        if not self.bm25:
            return []

        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        results = []

        for idx, score in enumerate(scores):
            if idx >= len(self.documents):
                continue

            doc = self.documents[idx]

            if company and doc["company"] != company:
                continue

            results.append(
                {
                    "score": float(score),
                    "company": doc["company"],
                    "source": doc["source"],
                    "text": doc["text"],
                }
            )

        results.sort(key=lambda x: x["score"], reverse=True)

        filtered = []

        for item in results:
            if item["score"] <= 0:
                continue

            filtered.append(item)

        return filtered[:top_k]