"""
Entity linker — discover entities shared across multiple documents.

Scans AKU extraction results to find entities appearing in 2+ documents,
filters out noise (short strings, numbers), and returns bridge candidates.
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Any

logger = logging.getLogger("indexrag.bridging_facts")


def find_bridge_entities(
    faq_data: List[Dict[str, Any]],
    min_entity_len: int = 4,
    min_doc_freq: int = 2,
    max_doc_freq: int = 10,
) -> Dict[str, Set[str]]:
    """
    Find entities that appear across multiple documents.

    Args:
        faq_data: List of FAQ extraction results, each with:
            - chunk_metadata.source: source document path
            - faqs[].entities: list of entity strings
        min_entity_len: Minimum entity string length.
        min_doc_freq: Minimum number of documents an entity must appear in.
        max_doc_freq: Maximum number of documents (filters overly generic entities).

    Returns:
        Dict mapping entity (lowercase) -> set of source document names.
    """
    # Map entity -> set of source documents
    entity_sources: Dict[str, Set[str]] = defaultdict(set)

    for item in faq_data:
        source = item.get("chunk_metadata", {}).get("source", "")
        source_name = Path(source).name if source else ""
        if not source_name:
            continue

        for faq in item.get("faqs", []):
            for ent in faq.get("entities", []):
                if isinstance(ent, str) and ent.strip():
                    entity_sources[ent.strip().lower()].add(source_name)

    # Filter
    filtered = {}
    for entity, sources in entity_sources.items():
        if len(entity) < min_entity_len:
            continue
        if entity.isdigit() or entity.startswith("$"):
            continue
        if entity.replace(",", "").replace(".", "").isdigit():
            continue
        if len(sources) < min_doc_freq or len(sources) > max_doc_freq:
            continue
        filtered[entity] = sources

    logger.info(
        f"Found {len(filtered)} bridge entities "
        f"(from {len(entity_sources)} total unique entities)"
    )
    return filtered


def collect_entity_facts(
    faq_items: List[Dict[str, Any]],
    entity_lower: str,
) -> List[str]:
    """Collect FAQ Q&A pairs that mention a given entity."""
    facts = []
    for faq in faq_items:
        q = faq.get("question", "").lower()
        a = faq.get("answer", "").lower()
        ents = [e.lower() for e in faq.get("entities", [])]
        if entity_lower in q or entity_lower in a or entity_lower in ents:
            facts.append(f"Q: {faq['question']}\nA: {faq['answer']}")
    return facts


def build_source_faq_index(
    faq_data: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Build mapping from source document name to list of FAQs."""
    source_faqs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in faq_data:
        source = item.get("chunk_metadata", {}).get("source", "")
        source_name = Path(source).name if source else ""
        if source_name:
            for faq in item.get("faqs", []):
                source_faqs[source_name].append(faq)
    return dict(source_faqs)
