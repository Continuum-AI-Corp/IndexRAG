"""
Build IndexRAG, Naive RAG, or GraphRAG knowledge bases.

Usage:
    # Build IndexRAG store (requires pre-extracted AKU cache)
    python -m scripts.build_kb --data-dir dataset/hotpotqa_1000_hf/documents \
        --kb-type indexrag --cache cache/hotpotqa_faqs.json

    # Build Naive RAG baseline
    python -m scripts.build_kb --data-dir dataset/hotpotqa_1000_hf/documents \
        --kb-type naive

    # Build GraphRAG baseline (requires AKU cache + pip install fast-graphrag)
    python -m scripts.build_kb --data-dir dataset/hotpotqa_1000_hf/documents \
        --kb-type graph --cache cache/hotpotqa_faqs.json

    # Build all
    python -m scripts.build_kb --data-dir dataset/hotpotqa_1000_hf/documents \
        --kb-type indexrag naive graph --cache cache/hotpotqa_faqs.json
"""

import json
import logging
from pathlib import Path

from indexrag.preprocessing import load_documents
from indexrag.indexing import build_indexrag_store, build_naive_store

logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")
logger = logging.getLogger("scripts.build_kb")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build IndexRAG knowledge bases")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory with .txt documents")
    parser.add_argument("--kb-type", type=str, nargs="+", default=["indexrag"],
                        choices=["indexrag", "naive", "graph"], help="KB type(s) to build")
    parser.add_argument("--suffix", type=str, default="", help="Suffix for KB directory name")
    parser.add_argument("--cache", type=str, default=None, help="Path to AKU extraction cache JSON")
    parser.add_argument("--bridging", type=str, default=None, help="Path to bridging facts JSON")
    parser.add_argument("--output-dir", type=str, default="vector_store", help="Root output directory")
    parser.add_argument("--chunk-size", type=int, default=None, help="Chunk size (default: 400 for naive)")
    parser.add_argument("--chunk-overlap", type=int, default=None, help="Chunk overlap (default: 80 for naive)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_root = Path(args.output_dir)

    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return

    for kb_type in args.kb_type:
        if kb_type == "naive":
            logger.info("Building Naive RAG store...")
            kb_path = output_root / f"conventional_vector{args.suffix}"
            documents = load_documents(data_dir)
            kwargs = {}
            if args.chunk_size is not None:
                kwargs["chunk_size"] = args.chunk_size
            if args.chunk_overlap is not None:
                kwargs["chunk_overlap"] = args.chunk_overlap
            build_naive_store(
                vector_store_dir=kb_path,
                documents=documents,
                **kwargs,
            )
            logger.info(f"Naive RAG store saved to {kb_path}")

        elif kb_type == "indexrag":
            if not args.cache:
                logger.error("--cache is required for indexrag KB type. Run scripts/extract_akus.py first.")
                continue

            cache_path = Path(args.cache)
            if not cache_path.exists():
                logger.error(f"Cache file not found: {cache_path}")
                continue

            logger.info("Building IndexRAG store...")
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)

            faq_data = cached.get("faq_data", [])

            # Load bridging facts if provided
            bridging_data = None
            if args.bridging:
                bridging_path = Path(args.bridging)
                if bridging_path.exists():
                    with open(bridging_path, "r", encoding="utf-8") as f:
                        bridging = json.load(f)
                    bridging_data = bridging.get("bridging_facts", [])
                    logger.info(f"Loaded {len(bridging_data)} bridging fact entries")

            kb_path = output_root / f"qa_vector_answer_augmented{args.suffix}"
            build_indexrag_store(
                vector_store_dir=kb_path,
                faq_data=faq_data,
                bridging_data=bridging_data,
            )
            logger.info(f"IndexRAG store saved to {kb_path}")

        elif kb_type == "graph":
            if not args.cache:
                logger.error("--cache is required for graph KB type. Run scripts/extract_akus.py first.")
                continue

            cache_path = Path(args.cache)
            if not cache_path.exists():
                logger.error(f"Cache file not found: {cache_path}")
                continue

            logger.info("Building GraphRAG store...")
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)

            faq_data = cached.get("faq_data", [])
            graph_dir = str(output_root / f"graph_store{args.suffix}")

            import asyncio
            from indexrag.indexing.graph_store import build_graph_store

            success = asyncio.run(build_graph_store(
                graph_dir=graph_dir,
                faq_data=faq_data,
            ))

            if success:
                logger.info(f"GraphRAG store saved to {graph_dir}")
            else:
                logger.error("Failed to build GraphRAG store")


if __name__ == "__main__":
    main()
