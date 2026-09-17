"""Section-aware chunking for SEC filings.

Naive fixed-size chunking splits mid-sentence across "Item" boundaries (Item 1A
Risk Factors, Item 7 MD&A, etc.), which hurts retrieval precision because a
chunk that starts three sentences into a risk factor loses its heading context.
This module first splits on the standard 10-K/10-Q "Item N." headings, then
sub-chunks each section by token count with overlap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Standard 10-K item headings (10-Q uses a similar but shorter set, matched by
# same regex since it's anchored on "Item <num>." pattern generically)
ITEM_HEADING_RE = re.compile(
    r"(?im)^\s*(item\s+\d+[a-z]?\.?\s*[-–—]?\s*[A-Z][A-Za-z ,&/]{2,80})\s*$"
)


@dataclass
class Chunk:
    text: str
    section: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


def split_into_sections(raw_text: str) -> list[tuple[str, str]]:
    """Splits filing text into (section_heading, section_text) pairs using
    Item N. headings. Falls back to a single 'Full Document' section if no
    headings are detected (some filers use non-standard formatting)."""
    matches = list(ITEM_HEADING_RE.finditer(raw_text))
    if not matches:
        return [("Full Document", raw_text)]

    sections: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        heading = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        body = raw_text[start:end].strip()
        if body:
            sections.append((heading, body))
    return sections


def _approx_token_count(text: str) -> int:
    # ~4 chars/token is a reasonable estimate for English prose without
    # pulling in a tokenizer dependency just for chunk sizing
    return max(1, len(text) // 4)


def chunk_section(
    section_heading: str, section_text: str, chunk_size_tokens: int, overlap_tokens: int
) -> list[Chunk]:
    words = section_text.split()
    chunk_size_words = chunk_size_tokens * 4 // 5  # rough tokens->words
    overlap_words = overlap_tokens * 4 // 5

    chunks: list[Chunk] = []
    i = 0
    idx = 0
    while i < len(words):
        window = words[i : i + chunk_size_words]
        text = " ".join(window)
        chunks.append(
            Chunk(
                text=f"[{section_heading}]\n{text}",
                section=section_heading,
                chunk_index=idx,
                metadata={"approx_tokens": _approx_token_count(text)},
            )
        )
        idx += 1
        if i + chunk_size_words >= len(words):
            break
        i += chunk_size_words - overlap_words
    return chunks


def chunk_filing(
    raw_text: str, chunk_size_tokens: int = 500, overlap_tokens: int = 75
) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for heading, body in split_into_sections(raw_text):
        all_chunks.extend(chunk_section(heading, body, chunk_size_tokens, overlap_tokens))
    return all_chunks
