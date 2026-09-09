"""
Custom AKU extraction example — use your own prompt template.

Shows how to:
1. Use a custom system prompt for AKU extraction
2. Process extraction results
3. Save results as a cache file for later KB building

Usage:
    python examples/custom_extraction.py --data-dir your_documents/ --prompt your_prompt.md
"""

import json
from pathlib import Path

from indexrag.preprocessing import load_documents, DocumentSplitter
from indexrag.aku import extract_akus_from_chunk


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Custom AKU extraction example")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--prompt", type=str, default=None,
                        help="Custom system prompt .md file (must contain {text} placeholder)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--output", type=str, default="custom_akus.json")
    args = parser.parse_args()

    # Load and chunk
    data_dir = Path(args.data_dir)
    documents = load_documents(data_dir)
    splitter = DocumentSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split(documents)
    print(f"Loaded {len(documents)} documents -> {len(chunks)} chunks")

    # Extract AKUs
    faq_data = []
    for i, chunk in enumerate(chunks):
        print(f"[{i+1}/{len(chunks)}] Extracting from {chunk.metadata.get('source', 'unknown')}...")
        result = extract_akus_from_chunk(
            chunk=chunk.page_content,
            chunk_id=str(i),
            source=chunk.metadata.get("source", ""),
            model=args.model,
            custom_prompt_file=args.prompt,
        )

        faq_data.append({
            "chunk_id": i,
            "chunk_text": chunk.page_content,
            "chunk_metadata": chunk.metadata,
            "faqs": [
                {"question": a.question, "answer": a.answer, "entities": a.entities}
                for a in result.akus
            ],
        })

        if result.error:
            print(f"  Warning: {result.error}")
        else:
            print(f"  Extracted {len(result.akus)} AKUs")

    # Save
    output = {
        "raw_docs": [{"page_content": d.page_content, "metadata": d.metadata} for d in documents],
        "chunked_docs": [{"page_content": c.page_content, "metadata": c.metadata} for c in chunks],
        "faq_data": faq_data,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    total_akus = sum(len(item["faqs"]) for item in faq_data)
    print(f"\nTotal AKUs: {total_akus}")
    print(f"Saved to: {args.output}")
    print(f"\nTo build a KB from this: python -m scripts.build_kb --kb-type indexrag --cache {args.output} --data-dir {args.data_dir}")


if __name__ == "__main__":
    main()
