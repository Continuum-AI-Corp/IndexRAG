"""Document loading utilities."""

import logging
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document

logger = logging.getLogger("indexrag.preprocessing")


def load_documents(
    doc_dir: Path,
    extensions: Optional[List[str]] = None,
) -> List[Document]:
    """
    Load documents from a directory.

    Args:
        doc_dir: Directory containing document files.
        extensions: File extensions to load (default: [".txt"]).

    Returns:
        List of LangChain Document objects.
    """
    extensions = extensions or [".txt"]
    documents = []

    for ext in extensions:
        for file_path in sorted(doc_dir.glob(f"*{ext}")):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if content:
                    documents.append(
                        Document(
                            page_content=content,
                            metadata={
                                "source": str(file_path),
                                "filename": file_path.name,
                            },
                        )
                    )
            except Exception as e:
                logger.warning(f"Failed to read {file_path}: {e}")

    logger.info(f"Loaded {len(documents)} documents from {doc_dir}")
    return documents
