"""
Prepare 2WikiMultihopQA dataset from HuggingFace.

Usage:
    python -m benchmarks.prepare_2wiki --output dataset/2wikimultihopqa_1000 --max-queries 1000
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict


def download_2wiki(split: str = "train", max_queries: int = 1000):
    """Download 2WikiMultihopQA from HuggingFace."""
    from datasets import load_dataset

    print(f"Downloading 2WikiMultihopQA ({split} split) from HuggingFace...")
    dataset = load_dataset("framolfese/2WikiMultihopQA", split=split)
    data = [dict(sample) for sample in dataset]
    print(f"Downloaded {len(data)} samples")
    return data[:max_queries]


def load_local(file_path: str) -> List[Dict[str, Any]]:
    """Load from local JSON/JSONL file."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if content.startswith("["):
            data = json.loads(content)
        else:
            f.seek(0)
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    return data


def prepare_dataset(data: List[Dict[str, Any]], output_dir: Path):
    """Convert to IndexRAG benchmark format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    documents_dir = output_dir / "documents"
    documents_dir.mkdir(exist_ok=True)

    questions = []
    ground_truth = {}

    # Collect unique documents
    documents = defaultdict(list)

    for i, sample in enumerate(data):
        sample_id = sample.get("id", sample.get("_id", f"2wiki_{i}"))

        questions.append({
            "id": sample_id,
            "question": sample["question"],
            "expected_answer": sample.get("answer", ""),
            "type": sample.get("type", "unknown"),
        })

        ground_truth[sample_id] = {
            "answer": sample.get("answer", ""),
            "type": sample.get("type", ""),
            "supporting_facts": sample.get("supporting_facts", {}),
        }

        # Extract documents from context
        context = sample.get("context", {})
        if isinstance(context, dict):
            titles = context.get("title", [])
            sentences_list = context.get("sentences", [])
            for j, title in enumerate(titles):
                if title not in documents and j < len(sentences_list):
                    documents[title] = sentences_list[j]
        elif isinstance(context, list):
            for title, sentences in context:
                if title not in documents:
                    documents[title] = sentences

    # Write documents
    for title, sentences in documents.items():
        safe_title = "".join(c if c.isalnum() or c in " _" else "_" for c in title)
        safe_title = safe_title.replace(" ", "_")[:100]
        doc_path = documents_dir / f"{safe_title}.txt"

        with open(doc_path, "w", encoding="utf-8") as f:
            f.write(f"# {title}\n\n")
            for sentence in sentences:
                f.write(f"{sentence}\n\n")

    # Save
    with open(output_dir / "questions.json", "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    with open(output_dir / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, ensure_ascii=False, indent=2)

    print(f"Queries: {len(questions)}")
    print(f"Documents: {len(documents)}")
    print(f"Output: {output_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Prepare 2WikiMultihopQA dataset")
    parser.add_argument("--input", type=str, default=None, help="Local JSON file (downloads from HF if not provided)")
    parser.add_argument("--output", type=str, default="dataset/2wikimultihopqa_1000")
    parser.add_argument("--max-queries", type=int, default=1000)
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()

    if args.input:
        data = load_local(args.input)[:args.max_queries]
    else:
        data = download_2wiki(args.split, args.max_queries)

    prepare_dataset(data, Path(args.output))


if __name__ == "__main__":
    main()
