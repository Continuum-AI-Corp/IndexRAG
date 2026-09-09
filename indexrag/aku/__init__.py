"""
AKU (Atomic Knowledge Unit) extraction module.

Extracts structured Q&A pairs from documents using LLM.
Each AKU contains: question, answer, entities.
"""

from .extractor import extract_akus_from_chunk, extract_summary_from_chunk
from .schema import AKU, AKUExtractionResult
