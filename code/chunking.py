import re
from typing import List


MIN_CHUNK_SIZE = 300
MAX_CHUNK_SIZE = 1000


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\t+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ ]{2,}", " ", text)

    return text.strip()


def split_oversized_paragraph(paragraph: str) -> List[str]:
    paragraph = paragraph.strip()

    if len(paragraph) <= MAX_CHUNK_SIZE:
        return [paragraph]

    sentences = re.split(r"(?<=[.!?])\s+", paragraph)

    chunks = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        candidate = f"{current} {sentence}".strip()

        if len(candidate) <= MAX_CHUNK_SIZE:
            current = candidate
        else:
            if current.strip():
                chunks.append(current.strip())

            current = sentence

    if current.strip():
        chunks.append(current.strip())

    return chunks


def merge_small_chunks(chunks: List[str]) -> List[str]:
    merged = []

    buffer = ""

    for chunk in chunks:
        chunk = chunk.strip()

        if not chunk:
            continue

        if len(chunk) >= MIN_CHUNK_SIZE:
            if buffer:
                combined = f"{buffer}\n\n{chunk}".strip()

                if len(combined) <= MAX_CHUNK_SIZE:
                    merged.append(combined)
                    buffer = ""
                    continue

                merged.append(buffer.strip())
                buffer = ""

            merged.append(chunk)
            continue

        candidate = f"{buffer}\n\n{chunk}".strip()

        if len(candidate) <= MAX_CHUNK_SIZE:
            buffer = candidate
        else:
            if buffer:
                merged.append(buffer.strip())

            buffer = chunk

    if buffer.strip():
        merged.append(buffer.strip())

    return merged


def split_markdown_chunks(content: str) -> List[str]:
    content = normalize_whitespace(content)

    if not content:
        return []

    sections = re.split(r"\n(?=#{1,6}\s)", content)

    raw_chunks = []
    seen = set()

    for section in sections:
        section = section.strip()

        if not section:
            continue

        paragraphs = re.split(r"\n\s*\n", section)

        current = ""

        for paragraph in paragraphs:
            paragraph = paragraph.strip()

            if not paragraph:
                continue

            oversized_parts = split_oversized_paragraph(paragraph)

            for part in oversized_parts:
                candidate = f"{current}\n\n{part}".strip()

                if len(candidate) <= MAX_CHUNK_SIZE:
                    current = candidate
                else:
                    if current.strip():
                        normalized = current.strip()

                        if normalized not in seen:
                            seen.add(normalized)
                            raw_chunks.append(normalized)

                    current = part

        if current.strip():
            normalized = current.strip()

            if normalized not in seen:
                seen.add(normalized)
                raw_chunks.append(normalized)

    final_chunks = merge_small_chunks(raw_chunks)

    cleaned = []

    seen_final = set()

    for chunk in final_chunks:
        normalized = normalize_whitespace(chunk)

        if (
            normalized
            and normalized not in seen_final
            and len(normalized) >= MIN_CHUNK_SIZE
        ):
            seen_final.add(normalized)
            cleaned.append(normalized)

    return cleaned