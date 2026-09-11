# Optional OrcaRouter integration

This branch adds an optional provider adapter for embeddings and LLM calls,
with API-key authentication or a local OAuth + PKCE login. It does not introduce
an IndexRAG method variant or new benchmark results.

## Research use and reproducibility

The paper overview, citation, reported results, prompts, dataset preparation,
and scoring metrics remain those of the original repository. With no provider
or model overrides, the code retains OpenAI with `gpt-4o-mini` for LLM calls
and `text-embedding-3-small` for embeddings.

OrcaRouter is opt-in. Switching the gateway or model is a separate experimental
configuration; the paper's reported scores are not measurements of this adapter
or of DeepSeek. For comparisons, record the provider, exact model, generation
settings, dataset split, and retrieval settings, and keep them consistent across
baselines. Use separate output/cache/index directories when changing models:
existing extracted AKUs and bridging facts also depend on the generating LLM.
The OrcaRouter embedding chunking and normalization described below apply only
to that provider and should be recorded as part of the configuration.

To use the repository's default OpenAI configuration in a shell previously
configured for this integration:

```bash
unset INDEXRAG_LLM_PROVIDER INDEXRAG_LLM_MODEL
unset INDEXRAG_EMBEDDING_PROVIDER INDEXRAG_EMBEDDING_MODEL
# Set your OpenAI key through your normal environment configuration.
# Remove any OPENAI_BASE_URL override if you want the standard OpenAI endpoint.
```

This restores provider/model defaults; it is not a substitute for following the
paper's complete experimental setup. Existing PKCE credentials are unused when
the OpenAI provider is selected.

## Setup

Install the project as described in the [README](../README.md#-installation).
Run the examples below from the repository root. Authentication is shared by
the two independently selectable providers.

## OrcaRouter embeddings: API key or PKCE login

OrcaRouter can provide embeddings for IndexRAG stores, Naive RAG stores and
semantic/hybrid retrieval. Choose either authentication method:

```bash
# Option 1: use your existing API key
export ORCAROUTER_API_KEY=sk-orca-...

# Option 2: sign in with your OrcaRouter account using OAuth + PKCE
indexrag-auth login
# To open the displayed URL yourself on the same machine:
indexrag-auth login --no-browser
```

The login command opens the OrcaRouter consent page and listens on a random
port bound to `127.0.0.1`. Approve IndexRAG; the browser redirects back automatically.
After the key is saved, the callback page attempts to close its tab automatically;
if the browser blocks this, it displays a manual-close message.
The callback verifies the per-login state before exchanging the code using S256
PKCE, without a client secret. Open the browser on the same machine as the CLI
(or arrange an SSH forward for the displayed callback port). This avoids the
unsupported `callback_url=oob` flow. Login expires after ten minutes; a failed/canceled login leaves
your previous credentials unchanged. The resulting key is stored locally at
`$XDG_CONFIG_HOME/indexrag/orcarouter.json` (default
`~/.config/indexrag/orcarouter.json`) with owner-only file permissions. This is
local credential storage, not an encrypted system keychain. An explicit
`ORCAROUTER_API_KEY` takes precedence over the saved login.

Enable the provider in every indexing and query process:

```bash
export INDEXRAG_EMBEDDING_PROVIDER=orcarouter
# Optional; this is the OrcaRouter default embedding model:
export INDEXRAG_EMBEDDING_MODEL=openai/text-embedding-3-small

# Embedding-only example: build a Naive RAG store from .txt documents
python -m scripts.build_kb --data-dir path/to/documents --kb-type naive
```

```python
from indexrag.retrieval import SemanticSearch

search = SemanticSearch(embedding_provider="orcarouter")
search.load_vector_store("vector_store/conventional_vector")
for document, distance in search.search("What connects these documents?", top_k=3):
    print(document.page_content, distance)
```

Oversized OrcaRouter inputs are split into Unicode-safe raw-text chunks (up to
8,000 UTF-8 bytes for OpenAI models, 2,000 for other model IDs). Chunk vectors
are averaged using UTF-8 byte lengths as weights, then normalized to produce
one vector per original document or query. Short-input vectors are normalized
as well, so FAISS compares long and short inputs on the same scale. Rebuild
indexes created with earlier versions of this integration if your embedding
model returns non-unit vectors. Requests contain at most 32 chunks to bound
aggregate request size;
both synchronous and asynchronous embedding methods use this behavior.

Use the same provider and embedding model when building and loading an index.
Rebuild existing indexes when changing embedding models; equal vector dimensions
do not imply compatible embedding spaces. The `embedding_provider` and
`embedding_model` keyword arguments also work with `build_indexrag_store` and
`build_naive_store`. Existing calls default to OpenAI.

`indexrag-auth status` reports whether credentials are configured without
printing them. `indexrag-auth logout` removes the local login; revoke the key in
your OrcaRouter console to invalidate it remotely. Environment keys are not
removed by logout. Starting another login or running logout invalidates older
pending login attempts. A cross-process lock protects the generation check and
credential replacement, so a delayed callback cannot restore a logged-out key
or overwrite a more recent login.

The defaults are `https://api.orcarouter.ai/v1` for embeddings and
`https://www.orcarouter.ai` for authorization. Override them independently with
`ORCA_API_BASE_URL` and `ORCA_AUTH_BASE_URL`, or use `ORCA_BASE_URL` as a shared
fallback. Saved keys are bound to the API URL used at login; changing that URL
requires a new login or an explicitly configured key.

## OrcaRouter LLM: extraction, bridging and answers

The same API key or saved PKCE login can also power AKU extraction, custom
summaries, bridging-fact generation and benchmark answer generation. Enable
LLM and embedding providers independently:

```bash
# Reuse your existing login; no new login is needed.
export INDEXRAG_LLM_PROVIDER=orcarouter
export INDEXRAG_LLM_MODEL=deepseek/deepseek-v4.1-flash
export INDEXRAG_EMBEDDING_PROVIDER=orcarouter
export INDEXRAG_EMBEDDING_MODEL=openai/text-embedding-3-small

# Extract AKUs, embed them, and retrieve with one OrcaRouter account:
python -m examples.quickstart --data-dir path/to/documents

# Or run individual stages:
python -m scripts.extract_akus --data-dir path/to/documents
python -m scripts.generate_bridging --cache cache/YOUR_CACHE_faqs.json
```

The default DeepSeek model uses non-thinking mode so reasoning does not consume
the output budget for JSON extraction and short answers. Empty answer text is
reported as an error.

OrcaRouter defaults to `deepseek/deepseek-v4.1-flash` for chat and
`openai/text-embedding-3-small` for embeddings. An explicit function `model`
argument or CLI `--model` / `--llm-model` overrides `INDEXRAG_LLM_MODEL`.
Without `INDEXRAG_LLM_PROVIDER=orcarouter`, LLM calls retain OpenAI credentials
and the `gpt-4o-mini` default. Selecting one provider does not silently change
the other. Benchmark vector retrieval now shares the same embedding factory
as index construction, so export the same embedding settings for evaluation.

To generate an answer after retrieval:

```python
from indexrag.retrieval import SemanticSearch
from benchmarks.evaluate import generate_answer

question = "What connects these documents?"
search = SemanticSearch()
search.load_vector_store("vector_store/quickstart_indexrag")
hits = search.search(question, top_k=3)
context = "\n\n".join(document.page_content for document, _ in hits)
print(generate_answer(context, question))
```

The optional GraphRAG backend keeps its own provider configuration. `.env` is
an example configuration file; export variables into the shell (the scripts
do not automatically load it).

