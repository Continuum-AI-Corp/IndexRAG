"""
Prepare MuSiQue dataset for IndexRAG benchmarks.

MuSiQue is a multi-hop QA dataset requiring 2-4 reasoning steps.
Each question comes with decomposed sub-questions and supporting paragraphs.

Usage:
    python -m benchmarks.prepare_musique --input musique_dev_1000.json --output dataset/musique_1000
"""

import json
from pathlib import Path
from typing import List, Dict, Any


def load_musique(file_path: str) -> List[Dict[str, Any]]:
    """Load MuSiQue samples from JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "data" in data:
        return data["data"]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Unknown format in {file_path}")


def prepare_dataset(
    samples: List[Dict[str, Any]], output_dir: Path, max_queries: int = 1000
):
    """Convert MuSiQue samples to IndexRAG benchmark format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    documents_dir = output_dir / "documents"
    documents_dir.mkdir(exist_ok=True)

    # Filter answerable only
    answerable = [s for s in samples if s.get("answerable", True)]
    selected = answerable[:max_queries]

    questions = []
    ground_truth = {}
    documents = {}  # title -> text (keep longest)

    for i, sample in enumerate(selected):
        sample_id = sample.get("id", f"musique_{i}")

        questions.append({
            "id": sample_id,
            "question": sample["question"],
            "expected_answer": sample.get("answer", ""),
            "num_hops": len(sample.get("question_decomposition", [])),
        })

        ground_truth[sample_id] = {
            "answer": sample.get("answer", ""),
            "answer_aliases": sample.get("answer_aliases", []),
            "num_hops": len(sample.get("question_decomposition", [])),
        }

        # Collect documents from paragraphs
        for para in sample.get("paragraphs", []):
            title = para.get("title", "unknown")
            text = para.get("paragraph_text", "")
            if title and text:
                if title not in documents or len(text) > len(documents[title]):
                    documents[title] = text

    # Write documents
    for title, text in documents.items():
        safe_title = "".join(c if c.isalnum() or c in " _" else "_" for c in title)
        safe_title = safe_title.replace(" ", "_")[:100]
        doc_path = documents_dir / f"{safe_title}.txt"

        with open(doc_path, "w", encoding="utf-8") as f:
            f.write(f"# {title}\n\n{text}\n")

    # Save
    with open(output_dir / "questions.json", "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    with open(output_dir / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, ensure_ascii=False, indent=2)

    print(f"Queries: {len(questions)} (answerable)")
    print(f"Documents: {len(documents)}")
    print(f"Output: {output_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Prepare MuSiQue dataset")
    parser.add_argument("--input", type=str, required=True, help="Path to MuSiQue JSON file")
    parser.add_argument("--output", type=str, default="dataset/musique_1000")
    parser.add_argument("--max-queries", type=int, default=1000)
    args = parser.parse_args()

    samples = load_musique(args.input)
    prepare_dataset(samples, Path(args.output), args.max_queries)


if __name__ == "__main__":
    main()
