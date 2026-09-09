# Benchmarks

## Supported Datasets

| Dataset | Type | Questions | Source |
|---------|------|-----------|--------|
| HotpotQA | Multi-hop QA | 1000 | [HuggingFace](https://huggingface.co/datasets/hotpot_qa) |
| 2WikiMultihopQA | Multi-hop QA | 1000 | [HuggingFace](https://huggingface.co/datasets/THUDM/2WikiMultihopQA) |
| MuSiQue | Multi-hop QA | 1000 | [GitHub](https://github.com/StonyBrookNLP/musique) |

## Preparing Datasets

```bash
# HotpotQA
python -m benchmarks.prepare_hotpotqa --output dataset/hotpotqa_1000_hf --max-queries 1000

# 2WikiMultihopQA
python -m benchmarks.prepare_2wiki --output dataset/2wikimultihopqa_1000 --max-queries 1000

# MuSiQue
python -m benchmarks.prepare_musique --output dataset/musique_1000 --max-queries 1000
```

## Running Evaluation

```bash
# IndexRAG
python -m benchmarks.evaluate --dataset hotpotqa_1000_hf --kb-type indexrag --top-k 5

# Naive RAG baseline
python -m benchmarks.evaluate --dataset hotpotqa_1000_hf --kb-type naive --top-k 5
```

## Metrics

- **EM (Exact Match)**: Normalized exact match between prediction and ground truth
- **F1**: Token-level F1 score
- **Substring Match**: Whether ground truth appears in prediction
