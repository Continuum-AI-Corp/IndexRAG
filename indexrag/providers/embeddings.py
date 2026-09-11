"""Shared embedding configuration for indexing and query-time retrieval."""

from __future__ import annotations

import os

from langchain_openai import OpenAIEmbeddings

from .length_safe import LengthSafeEmbeddings
from .orcarouter import api_key, api_url


def create_embeddings(model: str | None = None, provider: str | None = None, chunk_size: int = 64):
    provider = (provider or os.getenv("INDEXRAG_EMBEDDING_PROVIDER", "openai")).lower()
    model = (
        model
        or os.getenv("INDEXRAG_EMBEDDING_MODEL")
        or ("openai/text-embedding-3-small" if provider == "orcarouter" else "text-embedding-3-small")
    )
    if provider == "openai":
        return OpenAIEmbeddings(model=model, chunk_size=chunk_size)
    if provider != "orcarouter":
        raise ValueError("Embedding provider must be openai or orcarouter.")
    # Send raw text rather than OpenAI tokenizer IDs to the compatible gateway.
    return LengthSafeEmbeddings(
        OpenAIEmbeddings(
            model=model,
            api_key=api_key(),
            base_url=api_url(),
            chunk_size=chunk_size,
            check_embedding_ctx_length=False,
            model_kwargs={"encoding_format": "float"},
        )
    )
