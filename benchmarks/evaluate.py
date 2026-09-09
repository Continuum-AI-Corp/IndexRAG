"""
Evaluate IndexRAG and baselines on multi-hop QA benchmarks.

Supports: HotpotQA, 2WikiMultihopQA, MuSiQue.

Usage:
    python -m benchmarks.evaluate \\
        --dataset hotpotqa_1000_hf \\
        --kb-type indexrag \\
        --top-k 20 --context-docs 10
"""

import json
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any

from openai import OpenAI

from benchmarks.metrics import f1_score, substring_match, compute_metrics

logger = logging.getLogger("indexrag.benchmarks")

# Default number of context documents for answer generation.
# Even if top_k retrieves more, only this many are used for the LLM context.
CONTEXT_DOCS = 10

ANSWER_PROMPT = """You are a precise question answering assistant.

The provided context contains supporting information to help answer the question.
Use the context as your PRIMARY source, but you may also apply reasoning and your own knowledge if needed.

CRITICAL RULES:
1. Answer with ONLY the exact information requested
2. NO explanations, NO context, NO extra words
3. Be as concise as possible
4. If the answer is a name, give ONLY the name
5. If the answer is a number, give ONLY the number
6. If the answer is yes/no, give ONLY yes or no

EXAMPLES:
Q: "Who directed Old School?"
✓ GOOD: "Todd Phillips"
✗ BAD: "The director is Todd Phillips"

Q: "What year was it released?"
✓ GOOD: "2003"
✗ BAD: "The film was released in 2003"

Q: "Are they both American?"
✓ GOOD: "yes"
✗ BAD: "Yes, they are both American"

The context is supporting material - use it along with reasoning to provide the most accurate answer."""


def generate_answer(context: str, query: str, model: str = "gpt-4o-mini") -> str:
    """Generate answer using LLM given context and query."""
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": ANSWER_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer (be extremely concise):"},
        ],
        temperature=0,
        max_tokens=50,
    )
    return response.choices[0].message.content.strip()


def evaluate_vector_kb(
    kb_path: Path,
    queries: List[Dict[str, Any]],
    top_k: int = 5,
    context_docs: int = CONTEXT_DOCS,
    llm_model: str = "gpt-4o-mini",
    kb_type: str = "indexrag",
) -> Dict[str, Any]:
    """
    Evaluate a vector-based KB on a set of queries.

    Args:
        kb_path: Path to the FAISS vector store.
        queries: List of query dicts with 'question' and 'expected_answer'.
        top_k: Number of documents to retrieve.
        context_docs: Max number of retrieved docs used for LLM context.
        llm_model: Model for answer generation.
        kb_type: KB type ("indexrag" or "naive") — affects context separator.

    Returns:
        Dict with results and metrics.
    """
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.load_local(
        str(kb_path), embeddings, allow_dangerous_deserialization=True
    )

    results = []
    predictions = []
    ground_truths = []

    for i, q in enumerate(queries):
        question = q["question"]
        expected = q.get("expected_answer", "")

        # Retrieve top_k, use up to context_docs for answer generation
        docs = vectorstore.similarity_search_with_score(question, k=top_k)

        # Build context (capped at context_docs)
        context_parts = []
        for doc, score in docs:
            if len(context_parts) >= context_docs:
                break
            content = doc.metadata.get("merged_answers", doc.page_content)
            context_parts.append(content)
        # IndexRAG uses --- separator; naive uses plain double newline
        separator = "\n\n---\n\n" if kb_type == "indexrag" else "\n\n"
        context = separator.join(context_parts)

        # Generate answer
        predicted = generate_answer(context, question, model=llm_model)

        predictions.append(predicted)
        ground_truths.append(expected)

        results.append({
            "question": question,
            "predicted": predicted,
            "expected": expected,
            "f1": f1_score(predicted, expected),
            "substring_match": substring_match(predicted, expected),
        })

        if (i + 1) % 50 == 0:
            partial = compute_metrics(predictions, ground_truths)
            logger.info(f"  [{i+1}/{len(queries)}] F1={partial['f1']:.3f} SubMatch={partial['substring_match']:.3f}")

    metrics = compute_metrics(predictions, ground_truths)
    return {"results": results, "metrics": metrics}


def evaluate_graph_kb(
    graph_dir: str,
    queries: List[Dict[str, Any]],
    llm_model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """
    Evaluate a GraphRAG KB on a set of queries.

    Args:
        graph_dir: Path to the GraphRAG working directory.
        queries: List of query dicts with 'question' and 'expected_answer'.
        llm_model: Model for answer generation.

    Returns:
        Dict with results and metrics.
    """
    import asyncio
    from indexrag.retrieval.graph_search import load_graph, query_async

    graph = load_graph(
        dir_path=graph_dir,
        domain="general knowledge",
        entity_types=["person", "organization", "location", "event", "concept"],
        example_queries=["What is the relationship between X and Y?"],
    )

    if graph is None:
        print("Failed to load graph. Is fast-graphrag installed?")
        sys.exit(1)

    results = []
    predictions = []
    ground_truths = []

    for i, q in enumerate(queries):
        question = q["question"]
        expected = q.get("expected_answer", "")

        # Query graph for context (token limits aligned with original)
        context = asyncio.run(query_async(
            graph, question,
            only_context=True,
            entities_max_tokens=1000,
            relations_max_tokens=800,
            chunks_max_tokens=3200,
        ))
        context = str(context) if context else ""

        # Generate answer using same LLM pipeline
        predicted = generate_answer(context, question, model=llm_model) if context else ""

        predictions.append(predicted)
        ground_truths.append(expected)

        results.append({
            "question": question,
            "predicted": predicted,
            "expected": expected,
            "f1": f1_score(predicted, expected),
            "substring_match": substring_match(predicted, expected),
        })

        if (i + 1) % 50 == 0:
            partial = compute_metrics(predictions, ground_truths)
            logger.info(f"  [{i+1}/{len(queries)}] F1={partial['f1']:.3f} SubMatch={partial['substring_match']:.3f}")

    metrics = compute_metrics(predictions, ground_truths)
    return {"results": results, "metrics": metrics}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate IndexRAG on benchmarks")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., hotpotqa_1000_hf)")
    parser.add_argument("--kb-type", type=str, default="indexrag",
                        choices=["indexrag", "naive", "graph"], help="KB type")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k retrieval")
    parser.add_argument("--context-docs", type=int, default=CONTEXT_DOCS,
                        help="Max docs used for LLM context (default: 10)")
    parser.add_argument("--llm-model", type=str, default="gpt-4o-mini", help="LLM for answer generation")
    parser.add_argument("--kb-path", type=str, default=None, help="Path to KB directory (overrides auto-detection)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    # Load queries
    dataset_dir = Path(f"dataset/{args.dataset}")
    queries_path = dataset_dir / "questions.json"
    if not queries_path.exists():
        print(f"Queries not found at {queries_path}")
        sys.exit(1)

    with open(queries_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    # Determine KB path
    if args.kb_path:
        kb_path = args.kb_path
    elif args.kb_type == "indexrag":
        kb_path = "vector_store/qa_vector_answer_augmented"
    elif args.kb_type == "naive":
        kb_path = "vector_store/conventional_vector"
    elif args.kb_type == "graph":
        kb_path = "vector_store/graph_store"

    if not Path(kb_path).exists():
        print(f"KB not found at {kb_path}")
        sys.exit(1)

    print(f"Dataset: {args.dataset} ({len(queries)} queries)")
    print(f"KB: {args.kb_type} at {kb_path}")
    print(f"Top-k: {args.top_k}, Context docs: {args.context_docs}, LLM: {args.llm_model}")

    if args.kb_type == "graph":
        result = evaluate_graph_kb(
            graph_dir=kb_path,
            queries=queries,
            llm_model=args.llm_model,
        )
    else:
        result = evaluate_vector_kb(
            kb_path=Path(kb_path),
            queries=queries,
            top_k=args.top_k,
            context_docs=args.context_docs,
            llm_model=args.llm_model,
            kb_type=args.kb_type,
        )

    metrics = result["metrics"]
    print(f"\n{'='*60}")
    print(f"Results: EM={metrics['em']:.3f} F1={metrics['f1']:.3f} SubMatch={metrics['substring_match']:.3f}")
    print(f"{'='*60}")

    # Save
    output_path = args.output or f"test_results_{args.kb_type}_{args.dataset}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
