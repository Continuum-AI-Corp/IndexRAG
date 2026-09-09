"""
BM25 lexical search for keyword-based retrieval.

Complements semantic (embedding-based) search with exact term matching.
Supports Reciprocal Rank Fusion (RRF) for combining with embedding search.
"""

import pickle
import logging
from typing import List, Tuple, Dict, Any, Optional

logger = logging.getLogger("indexrag.retrieval")

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None


class BM25Index:
    """BM25 index for lexical search."""

    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.documents: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[List[str]] = []

    def build_index(
        self,
        documents: List[str],
        metadata: List[Dict[str, Any]],
    ):
        """Build BM25 index from documents."""
        if BM25Okapi is None:
            raise ImportError("rank-bm25 not installed. pip install rank-bm25")

        self.documents = documents
        self.metadata = metadata
        self.tokenized_corpus = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        logger.info(f"[BM25] Built index for {len(documents)} documents")

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search using BM25.

        Returns:
            List of (document, score, metadata) tuples, sorted by score (higher is better).
        """
        if self.bm25 is None:
            raise ValueError("BM25 index not built. Call build_index() first.")

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        return [
            (self.documents[i], scores[i], self.metadata[i]) for i in top_indices
        ]

    def save(self, file_path: str):
        """Save BM25 index to disk."""
        with open(file_path, "wb") as f:
            pickle.dump(
                {
                    "documents": self.documents,
                    "metadata": self.metadata,
                    "tokenized_corpus": self.tokenized_corpus,
                },
                f,
            )

    def load(self, file_path: str):
        """Load BM25 index from disk."""
        if BM25Okapi is None:
            raise ImportError("rank-bm25 not installed.")

        with open(file_path, "rb") as f:
            data = pickle.load(f)

        self.documents = data["documents"]
        self.metadata = data["metadata"]
        self.tokenized_corpus = data["tokenized_corpus"]
        self.bm25 = BM25Okapi(self.tokenized_corpus)


def reciprocal_rank_fusion(
    results_list: List[List[Tuple[Any, float]]],
    weights: Optional[List[float]] = None,
    k: int = 60,
) -> List[Tuple[Any, float]]:
    """
    Reciprocal Rank Fusion (RRF) for combining multiple ranked lists.

    Formula: RRF_score(d) = sum(weight_i / (k + rank_i(d)))
    """
    if not results_list:
        return []

    if weights is None:
        weights = [1.0 / len(results_list)] * len(results_list)

    total_weight = sum(weights)
    if total_weight > 0:
        weights = [w / total_weight for w in weights]

    rrf_scores: Dict[Any, float] = {}
    for result_list, weight in zip(results_list, weights):
        for rank, (item, _) in enumerate(result_list, start=1):
            rrf_score = weight / (k + rank)
            rrf_scores[item] = rrf_scores.get(item, 0) + rrf_score

    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


def hybrid_search_bm25_embedding(
    query: str,
    bm25_results: List[Tuple[str, float, Dict[str, Any]]],
    embedding_results: List[Tuple[str, float, Dict[str, Any]]],
    bm25_weight: float = 0.3,
    embedding_weight: float = 0.7,
    top_k: int = 5,
) -> List[Tuple[str, float, Dict[str, Any]]]:
    """
    Combine BM25 and embedding results using RRF.

    Args:
        bm25_results: (doc, score, metadata) from BM25.
        embedding_results: (doc, score, metadata) from embeddings.
        bm25_weight: Weight for BM25 results.
        embedding_weight: Weight for embedding results.
        top_k: Number of results to return.

    Returns:
        Fused results sorted by RRF score.
    """
    total = bm25_weight + embedding_weight
    if total > 0:
        bm25_weight /= total
        embedding_weight /= total

    doc_metadata_map: Dict[str, Dict[str, Any]] = {}
    for doc, _, meta in bm25_results:
        doc_metadata_map[doc] = meta
    for doc, _, meta in embedding_results:
        if doc not in doc_metadata_map:
            doc_metadata_map[doc] = meta

    bm25_for_rrf = [(doc, score) for doc, score, _ in bm25_results]
    embedding_for_rrf = [(doc, score) for doc, score, _ in embedding_results]

    fused = reciprocal_rank_fusion(
        [bm25_for_rrf, embedding_for_rrf],
        weights=[bm25_weight, embedding_weight],
    )

    return [
        (doc, rrf_score, doc_metadata_map.get(doc, {}))
        for doc, rrf_score in fused[:top_k]
    ]
