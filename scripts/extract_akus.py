"""
Extract AKUs (Atomic Knowledge Units) from documents with caching and concurrency.

Usage:
    python -m scripts.extract_akus --data-dir dataset/hotpotqa_1000_hf/documents
    python -m scripts.extract_akus --data-dir dataset/musique_1000/documents --suffix musique
"""

import json
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_text_splitters import RecursiveCharacterTextSplitter

from indexrag.preprocessing import load_documents
from indexrag.aku import extract_akus_from_chunk

CACHE_DIR = Path("cache")

shutdown_event = threading.Event()


def extract_single(args):
    """Extract AKUs from a single chunk."""
    chunk_id, content, metadata, model = args
    if shutdown_event.is_set():
        return {"chunk_id": chunk_id, "skipped": True}

    result = extract_akus_from_chunk(
        chunk=content,
        chunk_id=str(chunk_id),
        source=metadata.get("source", ""),
        model=model,
    )

    return {
        "chunk_id": chunk_id,
        "chunk_text": content,
        "chunk_metadata": metadata,
        "faqs": [
            {"question": a.question, "answer": a.answer, "entities": a.entities}
            for a in result.akus
        ],
        "success": result.error is None,
        "error": result.error,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extract AKUs from documents")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory with .txt documents")
    parser.add_argument("--suffix", type=str, default=None, help="Cache file suffix")
    parser.add_argument("--model", type=str, default=None, help="LLM model for extraction")
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=50)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    dataset_name = data_dir.parent.name
    suffix = f"_{args.suffix}" if args.suffix else ""
    cache_file = CACHE_DIR / f"{dataset_name}{suffix}_faqs.json"

    print(f"Data dir: {data_dir}")
    print(f"Cache file: {cache_file}")

    # Load existing cache
    existing_data = []
    existing_ids = set()

    if cache_file.exists():
        print(f"Loading existing cache...")
        with open(cache_file, "r", encoding="utf-8") as f:
            cached = json.load(f)
        existing_data = cached.get("faq_data", [])
        existing_ids = {item["chunk_id"] for item in existing_data}
        print(f"Found {len(existing_ids)} existing extractions")

    # Load and chunk documents
    print(f"Loading documents from {data_dir}...")
    raw_docs = load_documents(data_dir)
    print(f"Loaded {len(raw_docs)} documents")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        length_function=len,
    )
    chunked_docs = splitter.split_documents(raw_docs)
    print(f"Split into {len(chunked_docs)} chunks")

    # Prepare work
    chunks_to_process = [
        (i, doc.page_content, doc.metadata, args.model)
        for i, doc in enumerate(chunked_docs)
        if i not in existing_ids
    ]

    print(f"Chunks to process: {len(chunks_to_process)}")
    if not chunks_to_process:
        print("All chunks already extracted!")
        return

    # Extract with thread pool
    data_list = list(existing_data)
    stats = {"extracted": len(existing_ids), "errors": 0}
    completed = 0
    lock = threading.Lock()

    def save_progress():
        data_sorted = sorted(data_list, key=lambda x: x["chunk_id"])
        result = {
            "raw_docs": [{"page_content": d.page_content, "metadata": d.metadata} for d in raw_docs],
            "chunked_docs": [{"page_content": d.page_content, "metadata": d.metadata} for d in chunked_docs],
            "faq_data": data_sorted,
        }
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Extracting AKUs with {args.concurrency} workers...")

    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {executor.submit(extract_single, chunk): chunk[0]
                       for chunk in chunks_to_process}

            for future in as_completed(futures):
                if shutdown_event.is_set():
                    break

                result = future.result()
                if result.get("skipped"):
                    continue

                with lock:
                    data_list.append({
                        "chunk_id": result["chunk_id"],
                        "chunk_text": result["chunk_text"],
                        "chunk_metadata": result["chunk_metadata"],
                        "faqs": result["faqs"],
                    })

                    if result["success"]:
                        stats["extracted"] += 1
                    else:
                        stats["errors"] += 1

                    completed += 1
                    if completed % 50 == 0 or completed == len(chunks_to_process):
                        total_faqs = sum(len(item["faqs"]) for item in data_list)
                        print(f"  [{completed}/{len(chunks_to_process)}] Total AKUs: {total_faqs}")

                    if completed % args.save_interval == 0:
                        save_progress()

    except KeyboardInterrupt:
        print("\nInterrupted - saving progress...")
        shutdown_event.set()
        save_progress()
        print(f"Saved {len(data_list)} chunks. Run again to resume.")
        return

    # Final save
    save_progress()
    total_faqs = sum(len(item["faqs"]) for item in data_list)
    print(f"\nDone! Chunks: {len(data_list)}, AKUs: {total_faqs}, Errors: {stats['errors']}")
    print(f"Saved to: {cache_file}")


if __name__ == "__main__":
    main()
