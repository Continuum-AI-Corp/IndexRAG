"""
Naive RAG Store — baseline vector store using simple text chunks.

This is the conventional RAG baseline:
1. Load documents
2. Split into chunks
3. Store as vector embeddings (no FAQ extraction)
"""

import shutil
import logging
from pathlib import Path
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from ..retrieval.semantic_search import SemanticSearch

logger = logging.getLogger("indexrag.indexing")


def build_naive_store(
    vector_store_dir: Path,
    documents: Optional[List[Document]] = None,
    doc_dir: Optional[Path] = None,
    chunk_size: int = 400,
    chunk_overlap: int = 80,
) -> bool:
    """
    Build a naive RAG vector store (simple text chunks).

    Provide either pre-loaded `documents` or a `doc_dir` to load from.

    Args:
        vector_store_dir: Output directory for the FAISS vector store.
        documents: Pre-loaded LangChain documents.
        doc_dir: Directory containing .txt files (if documents not provided).
        chunk_size: Chunk size in characters.
        chunk_overlap: Overlap between chunks.

    Returns:
        True if successful.
    """
    logger.info(f"Building Naive RAG store (chunk_size={chunk_size})")

    if vector_store_dir.exists():
        shutil.rmtree(vector_store_dir)
    vector_store_dir.mkdir(parents=True, exist_ok=True)

    # Load documents if not provided
    if documents is None:
        if doc_dir is None:
            logger.error("Provide either documents or doc_dir")
            return False
        from ..preprocessing.loader import load_documents
        documents = load_documents(doc_dir)

    if not documents:
        logger.error("No documents to index")
        return False

    # Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunked_docs = splitter.split_documents(documents)
    logger.info(f"Split {len(documents)} docs into {len(chunked_docs)} chunks")

    # Store
    search = SemanticSearch(enable_bm25=True)
    search.create_vector_store(
        documents=chunked_docs,
        vector_store_path=str(vector_store_dir),
    )

    logger.info(f"Naive RAG store saved to {vector_store_dir}")
    return True
