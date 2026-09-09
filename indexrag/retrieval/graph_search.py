"""
Graph-based retrieval using fast-graphrag (optional dependency).

This module wraps fast-graphrag for graph-based knowledge retrieval.
If fast-graphrag is not installed, all functions gracefully return None.
"""

import logging
from typing import Any, List, Union, Optional

logger = logging.getLogger("indexrag.retrieval")

try:
    from fast_graphrag import GraphRAG, QueryParam
    HAS_GRAPHRAG = True
except ImportError:
    HAS_GRAPHRAG = False
    GraphRAG = None
    QueryParam = None


def load_graph(
    dir_path: str,
    domain: str,
    entity_types: List[str],
    example_queries: List[str],
) -> Optional[Any]:
    """Load or create a GraphRAG instance."""
    if not HAS_GRAPHRAG:
        logger.warning("fast-graphrag not installed. Graph search unavailable.")
        return None

    try:
        grag = GraphRAG(
            working_dir=dir_path,
            domain=domain,
            example_queries="\n".join(example_queries),
            entity_types=entity_types,
        )
        return grag
    except Exception as e:
        logger.error(f"Failed to load graph from {dir_path}: {e}")
        return None


async def insert_async(
    graph: Any,
    content: Union[str, List[str]],
    metadata: Optional[Any] = None,
) -> Optional[Any]:
    """Async insert content into graph."""
    if graph is None:
        return None
    try:
        return await graph.async_insert(content, metadata)
    except Exception as e:
        logger.error(f"Graph insert failed: {e}")
        return None


async def query_async(
    graph: Any,
    q: str,
    with_references: bool = False,
    only_context: bool = True,
    entities_max_tokens: int = 4000,
    relations_max_tokens: int = 3000,
    chunks_max_tokens: int = 9000,
) -> Optional[Any]:
    """Async query the graph."""
    if graph is None:
        return None

    try:
        await graph.state_manager.query_start()
        params = QueryParam(
            with_references=with_references,
            only_context=only_context,
            entities_max_tokens=entities_max_tokens,
            relations_max_tokens=relations_max_tokens,
            chunks_max_tokens=chunks_max_tokens,
        )
        answer = await graph.async_query(q, params)
        return answer
    except Exception as e:
        logger.error(f"Graph query failed for '{q}': {e}")
        return None
    finally:
        try:
            await graph.state_manager.query_done()
        except Exception:
            pass
