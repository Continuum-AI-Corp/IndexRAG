"""
IndexRAG Quickstart — end-to-end pipeline from documents to retrieval.

Prerequisites:
    export OPENAI_API_KEY=sk-...
    pip install -r requirements.txt

Usage:
    python examples/quickstart.py --data-dir dataset/hotpotqa_1000_hf/documents
"""

from pathlib import Path

from indexrag.preprocessing import load_documents, DocumentSplitter
from indexrag.aku import extract_akus_from_chunk
from indexrag.indexing import build_indexrag_store
from indexrag.retrieval import SemanticSearch


def main():
    import argparse

    parser = argparse.ArgumentParser(description="IndexRAG quickstart example")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory with .txt documents")
    parser.add_argument("--max-docs", type=int, default=5, help="Max documents to process (for demo)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    print(f"Loading documents from {data_dir}...")
    documents = load_documents(data_dir)[:args.max_docs]
    print(f"Loaded {len(documents)} documents")

    # Step 1: Chunk documents
    splitter = DocumentSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split(documents)
    print(f"Split into {len(chunks)} chunks")

    # Step 2: Extract AKUs
    print("\nExtracting AKUs...")
    faq_data = []
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}/{len(chunks)}...")
        result = extract_akus_from_chunk(
            chunk=chunk.page_content,
            chunk_id=str(i),
            source=chunk.metadata.get("source", ""),
        )
        faq_data.append({
            "chunk_metadata": chunk.metadata,
            "faqs": [
                {"question": a.question, "answer": a.answer, "entities": a.entities}
                for a in result.akus
            ],
        })
        print(f"    Extracted {len(result.akus)} AKUs")

    total_akus = sum(len(item["faqs"]) for item in faq_data)
    print(f"\nTotal AKUs: {total_akus}")

    # Step 3: Build IndexRAG store
    kb_path = Path("vector_store/quickstart_indexrag")
    print(f"\nBuilding IndexRAG store at {kb_path}...")
    build_indexrag_store(
        vector_store_dir=kb_path,
        faq_data=faq_data,
    )

    # Step 4: Query
    print("\nLoading store for retrieval...")
    search = SemanticSearch()
    search.load_vector_store(str(kb_path))

    query = "What is the main topic discussed in these documents?"
    print(f"\nQuery: {query}")
    results = search.search(query, top_k=3)

    print(f"\nTop {len(results)} results:")
    for i, (doc, score) in enumerate(results):
        content = doc.metadata.get("merged_answers", doc.page_content)[:200]
        print(f"  [{i+1}] score={score:.3f}")
        print(f"      {content}...")


if __name__ == "__main__":
    main()
