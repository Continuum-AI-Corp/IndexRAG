"""Data schemas for AKU extraction."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class AKU:
    """Atomic Knowledge Unit — a single question-answer-entities triple."""
    question: str
    answer: str
    entities: List[str] = field(default_factory=list)


@dataclass
class AKUExtractionResult:
    """Result of AKU extraction from a single chunk."""
    chunk_id: str
    source: str
    akus: List[AKU] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None and len(self.akus) > 0

    def to_faq_dicts(self) -> List[Dict[str, Any]]:
        """Convert to list of FAQ dicts (for backward compatibility)."""
        return [
            {"question": a.question, "answer": a.answer, "entities": a.entities}
            for a in self.akus
        ]
