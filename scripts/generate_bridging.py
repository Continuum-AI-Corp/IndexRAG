"""
Generate bridging facts from AKU extraction cache.

Usage:
    python -m scripts.generate_bridging --cache cache/hotpotqa_1000_hf_faqs.json
"""

import json
import logging
from pathlib import Path

from indexrag.bridging_facts import generate_all_bridging_facts

logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate bridging facts")
    parser.add_argument("--cache", type=str, required=True, help="Path to AKU extraction cache JSON")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path (default: auto)")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--min-entity-len", type=int, default=4)
    parser.add_argument("--min-doc-freq", type=int, default=2)
    parser.add_argument("--max-doc-freq", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=10)
    args = parser.parse_args()

    cache_path = Path(args.cache)
    if not cache_path.exists():
        print(f"Cache file not found: {cache_path}")
        return

    with open(cache_path, "r", encoding="utf-8") as f:
        cached = json.load(f)

    faq_data = cached.get("faq_data", [])
    print(f"Loaded {len(faq_data)} chunks from cache")

    output_path = Path(args.output) if args.output else cache_path.parent / f"{cache_path.stem}_bridging.json"

    result = generate_all_bridging_facts(
        faq_data=faq_data,
        output_path=output_path,
        min_entity_len=args.min_entity_len,
        min_doc_freq=args.min_doc_freq,
        max_doc_freq=args.max_doc_freq,
        concurrency=args.concurrency,
        model=args.model,
    )

    stats = result["stats"]
    print(f"\nEntities: {stats['completed_entities']}/{stats['total_entities']}")
    print(f"Bridging facts: {stats['total_bridging_facts']}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
