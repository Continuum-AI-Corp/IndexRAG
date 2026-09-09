"""
IndexRAG - A RAG framework using Atomic Knowledge Units and Bridging Facts.

Pipeline:
1. AKU Extraction: Extract atomic Q&A pairs from documents
2. Bridging Facts: Generate cross-document connections
3. Indexing: Build vector/graph knowledge bases
4. Retrieval: Semantic + BM25 hybrid search
"""

__version__ = "0.1.0"
