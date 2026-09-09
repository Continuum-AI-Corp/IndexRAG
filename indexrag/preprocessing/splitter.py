"""Document splitting using markdown-aware headers + recursive chunking."""

import logging
from typing import Dict, List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

logger = logging.getLogger("indexrag.preprocessing")


class DocumentSplitter:
    """Split documents into chunks, respecting markdown headers."""

    def __init__(
        self,
        headers_to_split_on: Optional[List[tuple]] = None,
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
    ):
        self.headers_to_split_on = headers_to_split_on or [
            ("#", "header_1"),
            ("##", "header_2"),
            ("###", "header_3"),
        ]
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on
        )
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n"],
            keep_separator=True,
        )

    def split(self, documents: List[Document]) -> List[Document]:
        """
        Split a list of LangChain Documents into smaller chunks.

        Args:
            documents: List of LangChain Document objects.

        Returns:
            List of chunked Document objects with preserved metadata.
        """
        return self.recursive_splitter.split_documents(documents)

    def split_text(self, text: str) -> List[Dict]:
        """
        Split raw text into chunks with markdown header awareness.

        Returns:
            List of dicts with 'content' and 'metadata' keys.
        """
        header_splits = self.header_splitter.split_text(text)
        final_splits = []

        for doc in header_splits:
            metadata = doc.metadata
            smaller_splits = self.recursive_splitter.split_text(doc.page_content)
            for chunk in smaller_splits:
                final_splits.append({"content": chunk, "metadata": {**metadata}})

        return final_splits

    def split_file(self, file_path: str) -> List[Dict]:
        """Split a markdown file into chunks."""
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return self.split_text(text)
