"""
Graph Store — build GraphRAG from AKU data (optional dependency).

Requires: pip install fast-graphrag
"""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger("indexrag.indexing")

try:
    from fast_graphrag import GraphRAG
    HAS_GRAPHRAG = True
except ImportError:
    HAS_GRAPHRAG = False


async def build_graph_store(
    graph_dir: str,
    faq_data: List[Dict[str, Any]],
    domain: str = "general knowledge",
    entity_types: Optional[List[str]] = None,
    example_queries: Optional[List[str]] = None,
) -> bool:
    """
    Build a GraphRAG store from AKU (FAQ) data.

    Args:
        graph_dir: Directory to store the graph.
        faq_data: List of AKU extraction results.
        domain: Domain description for GraphRAG.
        entity_types: Entity types for the graph.
        example_queries: Example queries for GraphRAG initialization.

    Returns:
        True if successful.
    """
    if not HAS_GRAPHRAG:
        logger.error(
            "fast-graphrag not installed. Install with: pip install fast-graphrag"
        )
        return False

    entity_types = entity_types or ["person", "organization", "location", "event", "concept"]
    example_queries = example_queries or ["What is the relationship between X and Y?"]

    try:
        grag = GraphRAG(
            working_dir=graph_dir,
            domain=domain,
            example_queries="\n".join(example_queries),
            entity_types=entity_types,
        )

        # Prepare content for insertion
        contents = []
        for item in faq_data:
            source = item.get("chunk_metadata", {}).get("source", "unknown")
            for faq in item.get("faqs", []):
                q = faq.get("question", "")
                a = faq.get("answer", "")
                if q and a:
                    contents.append(f"Q: {q}\nA: {a}")

        if not contents:
            logger.error("No FAQ content to insert into graph")
            return False

        logger.info(f"Inserting {len(contents)} AKU pairs into graph...")
        await grag.async_insert(contents)

        logger.info(f"Graph store built at {graph_dir}")
        return True

    except Exception as e:
        logger.error(f"Failed to build graph store: {e}")
        return False
