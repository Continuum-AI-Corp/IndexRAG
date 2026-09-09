"""
Prepare HotpotQA dataset from HuggingFace.

Downloads the distractor setting (which includes gold paragraphs)
and creates documents, queries, and ground truth files.

Usage:
    python -m benchmarks.prepare_hotpotqa --output dataset/hotpotqa_1000_hf --max-queries 1000
"""

import json
import random
from pathlib import Path


def download_hotpotqa(split: str = "validation", num_samples: int = 1000, seed: int = 42):
    """Download HotpotQA from HuggingFace and sample."""
    from datasets import load_dataset

    print(f"Downloading HotpotQA ({split} split) from HuggingFace...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    print(f"Total samples: {len(ds)}")

    random.seed(seed)
    indices = random.sample(range(len(ds)), min(num_samples, len(ds)))
    sampled = ds.select(indices)
    print(f"Sampled {len(sampled)} queries")

    return sampled


def prepare_dataset(sampled, output_dir: Path):
    """Convert HotpotQA samples to IndexRAG benchmark format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    documents_dir = output_dir / "documents"
    documents_dir.mkdir(exist_ok=True)

    questions = []
    ground_truth = {}
    saved_titles = set()
    doc_count = 0

    for entry in sampled:
        entry_id = entry["id"]

        # Query (flat format for evaluate.py)
        questions.append({
            "id": entry_id,
            "question": entry["question"],
            "expected_answer": entry["answer"],
            "type": entry["type"],
        })

        # Ground truth
        ground_truth[entry_id] = {
            "answer": entry["answer"],
            "type": entry["type"],
            "level": entry["level"],
            "supporting_facts": {
                "title": entry["supporting_facts"]["title"],
                "sent_id": entry["supporting_facts"]["sent_id"],
            },
        }

        # Create documents from context paragraphs
        titles = entry["context"]["title"]
        sentences_list = entry["context"]["sentences"]

        for title, sentences in zip(titles, sentences_list):
            if title in saved_titles:
                continue

            safe_title = "".join(c if c.isalnum() or c in "-_" else "_" for c in title)[:50]
            doc_path = documents_dir / f"{safe_title}.txt"

            counter = 1
            while doc_path.exists():
                doc_path = documents_dir / f"{safe_title}_{counter}.txt"
                counter += 1

            with open(doc_path, "w", encoding="utf-8") as f:
                f.write(f"# {title}\n\n")
                for sentence in sentences:
                    f.write(f"{sentence}\n\n")
            doc_count += 1
            saved_titles.add(title)

    # Save questions.json (flat list for evaluate.py)
    with open(output_dir / "questions.json", "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    # Save ground_truth.json
    with open(output_dir / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, ensure_ascii=False, indent=2)

    print(f"Queries: {len(questions)}")
    print(f"Documents: {doc_count}")
    print(f"Output: {output_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Prepare HotpotQA from HuggingFace")
    parser.add_argument("--output", type=str, default="dataset/hotpotqa_1000_hf")
    parser.add_argument("--max-queries", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="validation")
    args = parser.parse_args()

    sampled = download_hotpotqa(args.split, args.max_queries, args.seed)
    prepare_dataset(sampled, Path(args.output))


if __name__ == "__main__":
    main()
