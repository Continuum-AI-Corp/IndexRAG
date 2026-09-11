import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from indexrag.providers import orcarouter
from indexrag.providers.llm import chat_completion, resolve_llm_model


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for key in (
        "INDEXRAG_LLM_PROVIDER",
        "INDEXRAG_LLM_MODEL",
        "INDEXRAG_EMBEDDING_PROVIDER",
        "INDEXRAG_EMBEDDING_MODEL",
        "ORCAROUTER_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "ORCA_API_BASE_URL",
        "ORCA_AUTH_BASE_URL",
        "ORCA_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))


@pytest.fixture
def gateway(monkeypatch):
    calls, replies = [], []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append((self.path, self.headers.get("Authorization"), data))
            if self.path == "/v1/embeddings":
                result = {
                    "object": "list",
                    "model": data["model"],
                    "data": [
                        {"object": "embedding", "index": i, "embedding": [1.0, 0.0]} for i in range(len(data["input"]))
                    ],
                }
            else:
                assert self.path == "/v1/chat/completions"
                result = {
                    "id": "test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": data["model"],
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": replies.pop(0)},
                        }
                    ],
                }
            body = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("ORCA_API_BASE_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("INDEXRAG_LLM_PROVIDER", "orcarouter")
    monkeypatch.setenv("INDEXRAG_EMBEDDING_PROVIDER", "orcarouter")
    try:
        yield calls, replies
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("credential", ["api_key", "saved_login"])
def test_all_llm_paths_reuse_credentials(gateway, monkeypatch, tmp_path, credential):
    from benchmarks.evaluate import generate_answer
    from indexrag.aku.extractor import extract_akus_from_chunk, extract_summary_from_chunk
    from indexrag.bridging_facts.generator import generate_bridging_facts_llm

    calls, replies = gateway
    if credential == "api_key":
        monkeypatch.setenv("ORCAROUTER_API_KEY", "sk-orca-test")
    else:
        orcarouter.save_credential("sk-orca-test", "api")
    replies.extend(
        [
            json.dumps({"all_faqs": [{"question": "Capital?", "answer": "Paris", "entities": ["France"]}]}),
            "Paris is in France.",
            json.dumps(["France has Paris as its capital."]),
            "Paris",
        ]
    )
    result = extract_akus_from_chunk("Paris is the capital of France.")
    assert result.error is None
    assert result.akus[0].answer == "Paris"
    assert result.metadata["model"] == "deepseek/deepseek-v4.1-flash"
    prompt = tmp_path / "prompt.md"
    prompt.write_text("Summarize: {text}")
    assert extract_summary_from_chunk("Paris is in France.", str(prompt)) == "Paris is in France."
    assert generate_bridging_facts_llm("France", [("one", ["Paris is in France."]), ("two", ["France is in Europe."])])
    assert generate_answer("Paris is the capital of France.", "Capital?") == "Paris"
    assert len(calls) == 4
    assert all(c[1] == "Bearer sk-orca-test" for c in calls)
    assert all(c[2]["model"] == "deepseek/deepseek-v4.1-flash" for c in calls)
    assert all(c[2]["thinking"] == {"type": "disabled"} for c in calls)


def test_benchmark_uses_same_embedding_provider_as_index(gateway, monkeypatch, tmp_path):
    from benchmarks.evaluate import evaluate_vector_kb
    from indexrag.retrieval import SemanticSearch

    monkeypatch.setenv("ORCAROUTER_API_KEY", "sk-orca-test")
    calls, replies = gateway
    search = SemanticSearch(enable_bm25=False)
    search.create_vector_store([Document(page_content="Paris is the capital of France.")], str(tmp_path / "index"))
    replies.append("Paris")
    result = evaluate_vector_kb(tmp_path / "index", [{"question": "Capital of France?", "expected_answer": "Paris"}])
    assert result["metrics"]["f1"] == 1.0
    assert [c[0] for c in calls] == ["/v1/embeddings", "/v1/embeddings", "/v1/chat/completions"]
    assert calls[0][2]["model"] == calls[1][2]["model"] == "openai/text-embedding-3-small"


def test_model_override_and_openai_default(monkeypatch):
    from indexrag.providers import llm

    factory = MagicMock()
    monkeypatch.setattr(llm, "OpenAI", factory)
    assert resolve_llm_model() == "gpt-4o-mini"
    monkeypatch.setenv("INDEXRAG_LLM_MODEL", "env-model")
    chat_completion(messages=[{"role": "user", "content": "hello"}], model="explicit-model")
    factory.assert_called_once_with()
    create = factory.return_value.__enter__.return_value.chat.completions.create
    assert create.call_args.kwargs["model"] == "explicit-model"
    factory.return_value.__exit__.assert_called_once()
    assert resolve_llm_model() == "env-model"
    monkeypatch.setenv("INDEXRAG_LLM_PROVIDER", "typo")
    with pytest.raises(ValueError, match="provider"):
        chat_completion(messages=[])


def test_empty_answer_is_a_clear_failure(gateway, monkeypatch):
    monkeypatch.setenv("ORCAROUTER_API_KEY", "sk-orca-test")
    _, replies = gateway
    replies.append("")
    with pytest.raises(ValueError, match="no answer text"):
        chat_completion(messages=[{"role": "user", "content": "hello"}])


def test_explicit_thinking_configuration_is_preserved(gateway, monkeypatch):
    monkeypatch.setenv("ORCAROUTER_API_KEY", "sk-orca-test")
    calls, replies = gateway
    replies.append("answer")
    chat_completion(messages=[{"role": "user", "content": "hello"}], extra_body={"thinking": {"type": "enabled"}})
    assert calls[-1][2]["thinking"] == {"type": "enabled"}
