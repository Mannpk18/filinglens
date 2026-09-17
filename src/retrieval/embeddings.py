"""Thin wrapper around sentence-transformers so the rest of the codebase
depends on an interface, not a specific model. Swapping embedding models later
(e.g. for a cost/latency comparison writeup) means changing one line in .env.
"""

from __future__ import annotations

from functools import lru_cache

from src.config import settings


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = _get_model()
    # bge models recommend an instruction prefix for queries but not for
    # passages — see BAAI/bge-large-en-v1.5 model card
    return model.encode(texts, normalize_embeddings=True).tolist()


def embed_query(query: str) -> list[float]:
    instruction = "Represent this question for retrieving supporting evidence: "
    return embed_texts([instruction + query])[0]
