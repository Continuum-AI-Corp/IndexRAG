"""
Semantic search using FAISS vector store + OpenAI embeddings.

Supports optional BM25 hybrid search via Reciprocal Rank Fusion.
"""

import os
import logging
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from .bm25_search import BM25Index, hybrid_search_bm25_embedding

logger = logging.getLogger("indexrag.retrieval")


class SemanticSearch:
    """Vector-based semantic search with optional BM25 hybrid."""

    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        enable_bm25: bool = True,
    ):
        self.embeddings = OpenAIEmbeddings(model=embedding_model, chunk_size=2000)
        self.vector_store: Optional[FAISS] = None
        self.enable_bm25 = enable_bm25
        self.bm25_index: Optional[BM25Index] = BM25Index() if enable_bm25 else None

    def create_vector_store(
        self,
        documents: List[Document],
        vector_store_path: str,
    ) -> FAISS:
        """
        Create a FAISS vector store from documents.

        Args:
            documents: List of LangChain Document objects.
            vector_store_path: Path to save the vector store.

        Returns:
            FAISS vector store.
        """
        os.makedirs(os.path.dirname(vector_store_path) or ".", exist_ok=True)

        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        self.vector_store.save_local(vector_store_path)
        logger.info(
            f"Created vector store with {len(documents)} documents at {vector_store_path}"
        )

        # Build BM25 index alongside
        if self.enable_bm25 and self.bm25_index:
            texts = [doc.page_content for doc in documents]
            metadata = [doc.metadata for doc in documents]
            self.bm25_index.build_index(texts, metadata)
            bm25_path = f"{vector_store_path}_bm25.pkl"
            self.bm25_index.save(bm25_path)

        return self.vector_store

    def load_vector_store(
        self,
        path: str,
        allow_dangerous_deserialization: bool = True,
    ) -> FAISS:
        """Load a FAISS vector store from disk."""
        self.vector_store = FAISS.load_local(
            path, self.embeddings,
            allow_dangerous_deserialization=allow_dangerous_deserialization,
        )

        if self.enable_bm25 and self.bm25_index:
            bm25_path = f"{path}_bm25.pkl"
            if os.path.exists(bm25_path):
                self.bm25_index.load(bm25_path)
                logger.info(f"[BM25] Loaded index from {bm25_path}")

        return self.vector_store

    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents.

        Args:
            query: Search query.
            top_k: Number of results.
            score_threshold: Max distance (lower = more similar, 0 = no filter).

        Returns:
            List of (Document, score) tuples sorted by score (lower is better).
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call load_vector_store().")

        results = self.vector_store.similarity_search_with_score(query, k=top_k * 2)

        if score_threshold > 0:
            results = [(doc, score) for doc, score in results if score <= score_threshold]

        results.sort(key=lambda x: x[1])
        return results[:top_k]

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        bm25_weight: float = 0.3,
        embedding_weight: float = 0.7,
    ) -> List[Tuple[Document, float]]:
        """
        Hybrid search: BM25 (lexical) + Embedding (semantic) via RRF.

        Args:
            query: Search query.
            top_k: Number of results.
            bm25_weight: Weight for BM25 results.
            embedding_weight: Weight for embedding results.

        Returns:
            List of (Document, rrf_score) tuples.
        """
        if not self.enable_bm25 or not self.bm25_index or self.bm25_index.bm25 is None:
            logger.warning("BM25 not available, falling back to semantic-only search")
            return self.search(query, top_k)

        if self.vector_store is None:
            raise ValueError("Vector store not initialized.")

        # BM25 search
        bm25_results = self.bm25_index.search(query, top_k=top_k * 2)

        # Embedding search
        embedding_raw = self.vector_store.similarity_search_with_score(query, k=top_k * 2)
        embedding_results = [
            (doc.page_content, score, doc.metadata) for doc, score in embedding_raw
        ]

        # Fuse via RRF
        fused = hybrid_search_bm25_embedding(
            query=query,
            bm25_results=bm25_results,
            embedding_results=embedding_results,
            bm25_weight=bm25_weight,
            embedding_weight=embedding_weight,
            top_k=top_k,
        )

        # Convert back to (Document, score)
        return [
            (Document(page_content=doc, metadata=meta), score)
            for doc, score, meta in fused
        ]
