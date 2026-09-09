"""
Bridging Facts module — generate cross-document connections.

Identifies shared entities across documents and uses LLM to synthesize
facts that bridge information between multiple sources.
"""

from .entity_linker import find_bridge_entities
from .generator import generate_bridging_facts_llm
