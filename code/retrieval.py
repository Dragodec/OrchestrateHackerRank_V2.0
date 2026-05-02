import re
from collections import Counter
from typing import Dict, List, Optional

from rank_bm25 import BM25Okapi

from corpus import load_corpus


SUPPORTED_COMPANIES = {"hackerrank", "claude", "visa"}

TOP_K = 5
MAX_CONTEXT_CHARS = 3500


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

    return normalized.split()


def keyword_overlap(query_tokens, doc_tokens):
    return len(set(query_tokens) & set(doc_tokens))


class RetrievalEngine:
    def __init__(self, documents: List[Dict]):
        self.documents = documents or []

        self.tokenized_corpus = []

        for document in self.documents:
            combined = (
                f"{document.get('title', '')} "
                f"{document.get('content', '')}"
            )

            tokens = tokenize(combined)

            self.tokenized_corpus.append(
                tokens if tokens else ["empty"]
            )

        self.bm25 = (
            BM25Okapi(self.tokenized_corpus)
            if self.tokenized_corpus
            else None
        )

    def compute_confidence(
        self,
        selected: List[Dict],
    ) -> str:
        if not selected:
            return "low"

        top_score = selected[0]["score"]

        if len(selected) == 1:
            gap = top_score
        else:
            gap = top_score - selected[1]["score"]

        overlaps = [
            item.get("overlap", 0)
            for item in selected
        ]

        avg_overlap = sum(overlaps) / max(len(overlaps), 1)

        if top_score >= 12 and gap >= 3 and avg_overlap >= 3:
            return "high"

        if top_score >= 5 and avg_overlap >= 1:
            return "medium"

        return "low"

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

        query = normalize_text(
            f"{subject or ''} {issue or ''}"
        )

        query_tokens = tokenize(query)

        if not query_tokens:
            return {
                "results": [],
                "confidence": "low",
            }

        candidate_indices = []

        for index, doc in enumerate(self.documents):
            if (
                company in SUPPORTED_COMPANIES
                and doc["company"] != company
            ):
                continue

            candidate_indices.append(index)

        if not candidate_indices:
            return {
                "results": [],
                "confidence": "low",
            }

        bm25_scores = self.bm25.get_scores(query_tokens)

        scored = []

        query_counter = Counter(query_tokens)

        for index in candidate_indices:
            document = self.documents[index]

            base_score = float(bm25_scores[index])

            title_tokens = tokenize(document["title"])
            content_tokens = tokenize(document["content"])

            overlap = keyword_overlap(
                query_tokens,
                content_tokens,
            )

            title_overlap = keyword_overlap(
                query_tokens,
                title_tokens,
            )

            exact_phrase = (
                1
                if query in normalize_text(
                    document["content"]
                )
                else 0
            )

            density = 0

            if content_tokens:
                density = overlap / len(content_tokens)

            score = base_score

            score += overlap * 1.5
            score += title_overlap * 3.0
            score += exact_phrase * 4.0
            score += density * 10

            repeated_signal = sum(
                count
                for token, count in query_counter.items()
                if token in content_tokens
            )

            score += repeated_signal * 0.2

            scored.append(
                {
                    "score": round(score, 4),
                    "overlap": overlap,
                    "document": document,
                }
            )

        scored.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        selected = []

        seen_content = set()

        total_chars = 0

        for item in scored:
            document = item["document"]

            content = normalize_text(
                document["content"]
            )

            if content in seen_content:
                continue

            if total_chars >= MAX_CONTEXT_CHARS:
                break

            projected = (
                total_chars
                + len(document["content"])
            )

            if projected > MAX_CONTEXT_CHARS:
                continue

            seen_content.add(content)

            total_chars += len(document["content"])

            selected.append(
                {
                    "score": item["score"],
                    "overlap": item["overlap"],
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

        confidence = self.compute_confidence(selected)

        return {
            "results": selected,
            "confidence": confidence,
        }