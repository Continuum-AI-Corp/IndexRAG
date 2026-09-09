"""
IndexRAG Store — the main knowledge base combining AKU answers + bridging facts.

This is the primary IndexRAG indexing method:
- FAQ answer vectors (one per source chunk, merged answers)
- Bridging fact vectors (one per cross-document bridging fact)

Both are stored in a single FAISS vector store for unified retrieval.
"""

import shutil
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger("indexrag.indexing")


def build_indexrag_store(
    vector_store_dir: Path,
    faq_data: List[Dict[str, Any]],
    bridging_data: Optional[List[Dict[str, Any]]] = None,
    embedding_model: str = "text-embedding-3-small",
) -> bool:
    """
    Build the IndexRAG vector store (FAQ answers + bridging facts).

    Args:
        vector_store_dir: Output directory for the FAISS vector store.
        faq_data: List of AKU extraction results. Each item has:
            - chunk_metadata: {source, source_index}
            - faqs: [{question, answer, entities}, ...]
        bridging_data: Optional list of bridging fact results. Each item has:
            - entity: bridge entity name
            - sources: list of source document names
            - bridging_facts: list of fact strings
        embedding_model: OpenAI embedding model name.

    Returns:
        True if successful.
    """
    logger.info("Building IndexRAG store (AKU answers + bridging facts)")

    if vector_store_dir.exists():
        shutil.rmtree(vector_store_dir)
    vector_store_dir.mkdir(parents=True, exist_ok=True)

    if not faq_data:
        logger.error("No FAQ data provided!")
        return False

    # Part 1: FAQ answer documents (one per source chunk)
    faq_documents = []
    faq_count = 0

    for item in faq_data:
        source = item.get("chunk_metadata", {}).get("source", "unknown")
        chunk_index = item.get("chunk_metadata", {}).get("source_index", 0)

        merged_answers = []
        for faq in item.get("faqs", []):
            merged_answers.append(faq["answer"])
            faq_count += 1

        if not merged_answers:
            continue

        embedding_content = "\n\n".join(merged_answers)
        doc = Document(
            page_content=embedding_content,
            metadata={
                "type": "faq_answer_merged",
                "source": source,
                "chunk_index": chunk_index,
                "num_faqs": len(merged_answers),
                "merged_answers": embedding_content,
            },
        )
        faq_documents.append(doc)

    logger.info(f"FAQ documents: {len(faq_documents)} (from {faq_count} AKUs)")

    # Part 2: Bridging fact documents
    bridge_documents = []
    if bridging_data:
        for item in bridging_data:
            facts = item.get("bridging_facts", [])
            if not facts:
                continue

            entity = item.get("entity", "")
            sources = item.get("sources", [])
            source_label = "+".join(sources[:3]) if sources else "unknown"

            for fact in facts:
                if not isinstance(fact, str) or not fact.strip():
                    continue

                doc = Document(
                    page_content=fact,
                    metadata={
                        "type": "bridging_fact",
                        "entity": entity,
                        "source": f"bridge:{source_label}",
                        "merged_answers": fact,
                    },
                )
                bridge_documents.append(doc)

        logger.info(f"Bridging fact documents: {len(bridge_documents)}")

    # Part 3: Combine and build FAISS
    all_documents = faq_documents + bridge_documents
    logger.info(f"Total documents: {len(all_documents)}")

    embeddings = OpenAIEmbeddings(model=embedding_model)
    vector_store = FAISS.from_documents(all_documents, embeddings)
    vector_store.save_local(str(vector_store_dir))

    logger.info(f"IndexRAG store saved to {vector_store_dir}")
    return True
