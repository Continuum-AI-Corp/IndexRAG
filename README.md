<div align="center">

# IndexRAG

_Index-Time Reasoning for Multi-Hop Retrieval-Augmented Generation_

AACL-IJCNLP 2026, Findings

[![arXiv](https://img.shields.io/badge/arXiv-2603.16415-b31b1b.svg)](https://arxiv.org/abs/2603.16415)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-green.svg)](https://python.org)
[![Vector Store](https://img.shields.io/badge/Index-FAISS-orange.svg)](https://github.com/facebookresearch/faiss)

</div>

<p align="center">
  <img src="IndexRAG_Overview.png" width="780" alt="IndexRAG Overview">
</p>

## 🎯 Overview

**IndexRAG** shifts cross-document reasoning from online inference to offline
indexing. Graph-based and iterative RAG systems answer multi-hop questions by
doing extra work at query time — entity extraction, graph traversal, or several
rounds of retrieval and generation. IndexRAG does that work once, while
building the index, and stores the result as **bridging facts**: independently
retrievable units that already encode a cross-document connection.

At query time nothing special happens. One retrieval pass over a flat vector
index, one LLM call, done.

The repository covers:

- **Two offline stages**: atomic knowledge unit (AKU) extraction, then
  cross-document bridging fact generation
- **3 multi-hop QA benchmarks**: HotpotQA, 2WikiMultiHopQA, MuSiQue
- **4 index variants**: IndexRAG, AKU-only, Naive RAG chunks, graph-based
- **3 retrieval backends**: dense, BM25 with reciprocal rank fusion, graph
  traversal
- **Training-free**: no fine-tuning of the embedding model or the LLM

<p align="center">
    🔨&nbsp;<a href="#-installation">Installation</a>
    | 🚀&nbsp;<a href="#-quick-start">Quick Start</a>
    | ✨&nbsp;<a href="#-key-results">Results</a>
    | 📁&nbsp;<a href="#-project-structure">Structure</a>
    | 🔗&nbsp;<a href="#-citation">Citation</a>
</p>

## 🔗 Citation

```bibtex
@article{bao2026indexrag,
  title   = {IndexRAG: Bridging Facts for Cross-Document Reasoning at Index Time},
  author  = {Bao, Zhenghua and Shi, Yi},
  journal = {arXiv preprint arXiv:2603.16415},
  year    = {2026}
}
```

The Findings version is forthcoming. This entry will be replaced by the ACL
Anthology one once it is available.

## ✨ Key Results

F1 on 1,000 sampled questions per benchmark. All methods below use **a single
retrieval pass and a single LLM call** at inference time.

| Method | HotpotQA | 2WikiMultiHopQA | MuSiQue | Avg |
|--------|:--------:|:---------------:|:-------:|:---:|
| BM25 | 60.3 | 35.9 | 19.2 | 38.5 |
| Naive RAG | 63.6 | 47.7 | 29.9 | 47.1 |
| RAPTOR | 63.6 | 47.8 | 29.7 | 47.0 |
| FastGraphRAG | 63.5 | **57.4** | 27.2 | 49.4 |
| **IndexRAG** | **68.9** | 51.7 | **34.4** | **51.7** |

IndexRAG gains **+4.6 F1** over Naive RAG on average without adding an online
LLM call.

Methods that spend more at query time are a separate class. HippoRAG2 reaches
54.1 avg F1 using two LLM calls and graph traversal. Bridging facts compose
with those methods rather than competing with them: adding them to IRCoT
reaches **55.0 avg F1**.

The saving is on the online side. Against HippoRAG2, retrieval is
**6.6–10.4× faster**, on an index **17–20× smaller** and **30–50% cheaper** to
build.

## 🔨 Installation

### Prerequisites

- Python 3.9 or newer
- An OpenAI API key (used for extraction, bridging, embeddings and answering)

### Setup

```bash
git clone https://github.com/Continuum-AI-Corp/IndexRAG.git
cd IndexRAG

pip install -e .                    # core
pip install -e ".[benchmarks]"      # + dataset download tools
pip install -e ".[graph]"           # + graph-based baseline
```

Copy the example environment file and fill in your key:

```bash
cp .env.example .env
# or simply
export OPENAI_API_KEY=sk-...
```

### Optional: OrcaRouter PKCE login and LLM

This branch supports OrcaRouter for both LLM calls and embeddings. This is an
optional provider integration: the paper's results and experimental instructions
remain unchanged. Runs using a different model are separate experiments.

**1. Check out this branch and install the login command:**

```bash
git fetch origin
git switch feat/orcarouter-pkce-llm
pip install -e .
```

**2. Authenticate with PKCE (once):**

```bash
indexrag-auth login
indexrag-auth status
```

Approve the request in the browser on the same machine as the terminal. The
browser returns to a local callback and the command saves your credentials;
the success page attempts to close automatically. If no browser opens, run
`indexrag-auth login --no-browser` and open the displayed URL yourself.

Both LLM and embedding calls reuse this login. Credentials are saved at
`~/.config/indexrag/orcarouter.json` (or under `XDG_CONFIG_HOME`). Alternatively,
set `ORCAROUTER_API_KEY` instead of logging in; an environment key takes
precedence over the saved login. `indexrag-auth logout` removes the local login.

**3. Enable LLM and embeddings in your shell:**

```bash
export INDEXRAG_LLM_PROVIDER=orcarouter
export INDEXRAG_LLM_MODEL=deepseek/deepseek-v4.1-flash
export INDEXRAG_EMBEDDING_PROVIDER=orcarouter
export INDEXRAG_EMBEDDING_MODEL=openai/text-embedding-3-small
```

The LLM setting applies to AKU extraction, custom summaries, bridging facts and
benchmark answer generation. The two providers are independent: selecting only
the LLM provider leaves embeddings on OpenAI. With both set as above, these
vector-pipeline stages do not require an OpenAI key. Export the settings in each
new shell; the scripts do not automatically load `.env`.

**4. Extract knowledge, build an index and retrieve from your `.txt` documents:**

```bash
python -m examples.quickstart --data-dir path/to/documents --max-docs 5
```

This writes `vector_store/quickstart_indexrag`. To generate an LLM answer from
that index, run the following from the repository root in the same environment:

```python
from indexrag.retrieval import SemanticSearch
from benchmarks.evaluate import generate_answer

search = SemanticSearch()
search.load_vector_store("vector_store/quickstart_indexrag")
question = "What is the main topic discussed in these documents?"
hits = search.search(question, top_k=3)
context = "\n\n".join(document.page_content for document, _ in hits)
print(generate_answer(context, question))
```

Use matching embedding settings for indexing and retrieval, and separate
indexes/caches for different experimental configurations. Without provider
or model overrides, the original OpenAI defaults remain in effect. The optional
GraphRAG backend has its own provider configuration.
See the [full integration guide](docs/orcarouter.md) for endpoint overrides,
credential handling and reproducibility details.

## 🚀 Quick Start

The shortest path from a folder of documents to an answer:

```bash
python -m examples.quickstart --data-dir path/to/documents
```

The full pipeline is four offline steps and one query step.

### 1. Prepare a dataset

HotpotQA and 2WikiMultiHopQA download automatically. MuSiQue must be fetched
manually from [its repository](https://github.com/StonyBrookNLP/musique).

```bash
python -m benchmarks.prepare_hotpotqa --output dataset/hotpotqa_1000_hf --max-queries 1000
python -m benchmarks.prepare_2wiki    --output dataset/2wikimultihopqa_1000 --max-queries 1000
python -m benchmarks.prepare_musique  --input path/to/musique_ans_v1.0_dev.jsonl \
                                      --output dataset/musique_1000
```

### 2. Extract AKUs (Stage 1)

```bash
python -m scripts.extract_akus --data-dir dataset/hotpotqa_1000_hf/documents
```

Writes a cache of atomic question-answer pairs and their entities, one entry
per document.

### 3. Generate bridging facts (Stage 2)

```bash
python -m scripts.generate_bridging --cache cache/hotpotqa_1000_hf_faqs.json
```

Entities appearing in two or more documents become bridge entities. Each one
yields bridging facts that connect its source documents.

### 4. Build the index

```bash
# IndexRAG: AKUs and bridging facts in one flat store
python -m scripts.build_kb \
    --data-dir dataset/hotpotqa_1000_hf/documents \
    --kb-type indexrag \
    --cache cache/hotpotqa_1000_hf_faqs.json \
    --bridging cache/hotpotqa_1000_hf_faqs_bridging.json

# Naive RAG baseline
python -m scripts.build_kb --data-dir dataset/hotpotqa_1000_hf/documents --kb-type naive

# Graph baseline
python -m scripts.build_kb --data-dir dataset/hotpotqa_1000_hf/documents \
    --kb-type graph --cache cache/hotpotqa_1000_hf_faqs.json
```

### 5. Evaluate

```bash
python -m benchmarks.evaluate --dataset hotpotqa_1000_hf --kb-type indexrag --top-k 20 --context-docs 10
python -m benchmarks.evaluate --dataset hotpotqa_1000_hf --kb-type naive --top-k 10
python -m benchmarks.evaluate --dataset hotpotqa_1000_hf --kb-type graph
```

## 📁 Project Structure

```
indexrag/
    preprocessing/        document loading, fixed-size chunking
    aku/                  Stage 1: AKU extraction, schema, prompts
    bridging_facts/       Stage 2: entity linking, bridging fact generation
    indexing/             index construction for each variant
    retrieval/            dense, BM25 with RRF, graph traversal

scripts/                  offline pipeline entry points
benchmarks/               dataset preparation, metrics, evaluation driver
examples/                 quickstart and a custom Stage 1 strategy
```

## ⚙️ Configuration

Settings behind the reported results. Retrieval depth is passed explicitly, as in
the benchmark commands above:

| | Value |
|---|---|
| LLM (all stages) | `gpt-4o-mini` |
| Embeddings | `text-embedding-3-small` |
| Vector store | FAISS, flat index |
| Retrieval | top 20 by cosine similarity, 10 passed to the LLM |
| Bridge entity document frequency | 2 to 10 |
| Source documents per bridge entity | at most 5 |
| Facts per source document | at most 8 |
| Passage chunking (baselines) | about 100 words, 80 character overlap |

Prompt templates live next to the code that uses them, under
`indexrag/aku/prompts/`, and are kept verbatim so results stay reproducible.

## 📄 License

Apache-2.0. See [LICENSE](LICENSE).

Benchmark datasets keep their original licences. HotpotQA and MuSiQue are
CC BY-SA 4.0, 2WikiMultiHopQA is Apache-2.0. FAISS is MIT.

## 🙏 Contact

Questions and bug reports are best filed as
[issues](https://github.com/Continuum-AI-Corp/IndexRAG/issues).
For anything else, `research@orcarouter.ai`.
