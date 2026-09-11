import base64
import hashlib
import json
import stat
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from langchain_core.documents import Document

from indexrag.indexing import build_indexrag_store, build_naive_store
from indexrag.providers import create_embeddings, orcarouter
from indexrag.retrieval.semantic_search import SemanticSearch


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    for key in (
        "ORCAROUTER_API_KEY",
        "ORCA_BASE_URL",
        "ORCA_API_BASE_URL",
        "ORCA_AUTH_BASE_URL",
        "INDEXRAG_EMBEDDING_PROVIDER",
        "INDEXRAG_EMBEDDING_MODEL",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))


@pytest.fixture
def gateway(monkeypatch):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append((self.path, self.headers.get("Authorization"), data))
            assert self.path == "/v1/embeddings"
            assert isinstance(data["input"], list)
            assert all(isinstance(text, str) for text in data["input"])
            result = {
                "object": "list",
                "model": data["model"],
                "data": [
                    {"object": "embedding", "index": i, "embedding": [1.0, 0.0] if "apple" in text else [0.0, 1.0]}
                    for i, text in enumerate(data["input"])
                ],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
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
    monkeypatch.setenv("INDEXRAG_EMBEDDING_PROVIDER", "orcarouter")
    monkeypatch.setenv("INDEXRAG_EMBEDDING_MODEL", "test-embedding")
    try:
        yield calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_api_key_build_save_reload_query(gateway, monkeypatch, tmp_path):
    monkeypatch.setenv("ORCAROUTER_API_KEY", "sk-orca-explicit")
    search = SemanticSearch(enable_bm25=False)
    search.create_vector_store(
        [Document(page_content="apple fruit"), Document(page_content="ocean water")], str(tmp_path / "index")
    )
    reloaded = SemanticSearch(enable_bm25=False)
    reloaded.load_vector_store(str(tmp_path / "index"))
    assert reloaded.search("apple", top_k=1)[0][0].page_content == "apple fruit"
    assert all(c[1] == "Bearer sk-orca-explicit" for c in gateway)
    assert all(c[2]["model"] == "test-embedding" for c in gateway)
    assert all(c[2]["encoding_format"] == "float" for c in gateway)


def test_pkce_saved_key_drives_real_embedding_client(gateway, monkeypatch):
    verifier, url = orcarouter.begin_login("http://127.0.0.1:51733/cb", "test-state")
    query = parse_qs(urlsplit(url).query)
    assert query["callback_url"] == ["http://127.0.0.1:51733/cb"]
    assert query["code_challenge_method"] == ["S256"]
    assert verifier not in url
    assert orcarouter.begin_login("http://127.0.0.1:51733/cb", "test-state")[0] != verifier

    def exchange(url, *, json, **kwargs):
        assert url == "https://www.orcarouter.ai/api/v1/auth/keys"
        challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(json["code_verifier"].encode()).digest()).rstrip(b"=").decode()
        )
        assert challenge == query["code_challenge"][0]
        assert json["code"] == "one-time-code"
        assert kwargs["follow_redirects"] is False
        return httpx.Response(200, json={"key": "sk-orca-oauth", "scope": "api"})

    monkeypatch.setattr(orcarouter.httpx, "post", exchange)
    orcarouter.exchange_code("one-time-code", verifier)
    assert stat.S_IMODE(orcarouter.credential_path().stat().st_mode) == 0o600
    assert create_embeddings().embed_query("apple") == [1.0, 0.0]
    assert gateway[-1][1] == "Bearer sk-orca-oauth"


def test_all_store_builders_use_provider(gateway, monkeypatch, tmp_path):
    monkeypatch.setenv("ORCAROUTER_API_KEY", "sk-orca-build")
    assert build_naive_store(tmp_path / "naive", documents=[Document(page_content="apple fruit")])
    assert build_indexrag_store(
        tmp_path / "aku",
        faq_data=[
            {
                "chunk_metadata": {"source": "test", "source_index": 0},
                "faqs": [{"question": "fruit?", "answer": "apple fruit", "entities": ["apple"]}],
            }
        ],
    )
    for name in ("naive", "aku"):
        search = SemanticSearch(enable_bm25=False)
        search.load_vector_store(str(tmp_path / name))
        assert "apple" in search.search("apple", top_k=1)[0][0].page_content
    assert len(gateway) == 4


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(403, json={"key": "secret-response"}),
        httpx.Response(200, json={"key": "sk-orca-secret", "scope": "connector"}),
        httpx.Response(200, json={"scope": "api"}),
        httpx.Response(200, text="not json"),
    ],
)
def test_failed_login_preserves_credentials(monkeypatch, response):
    orcarouter.save_credential("sk-orca-old", "api")
    monkeypatch.setattr(orcarouter.httpx, "post", lambda *args, **kwargs: response)
    with pytest.raises(ValueError) as exc:
        orcarouter.exchange_code("bad", "verifier")
    assert "sk-orca-secret" not in str(exc.value)
    assert orcarouter.api_key() == "sk-orca-old"


def test_credentials_bound_to_endpoint_and_env_precedence(monkeypatch):
    orcarouter.save_credential("sk-orca-saved", "api")
    monkeypatch.setenv("ORCA_API_BASE_URL", "https://other.example/v1")
    with pytest.raises(ValueError):
        orcarouter.api_key()
    monkeypatch.setenv("ORCAROUTER_API_KEY", "sk-orca-explicit")
    assert orcarouter.api_key() == "sk-orca-explicit"
    assert orcarouter.api_url() == "https://other.example/v1"


def test_provider_fails_closed_and_openai_still_default(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test")
    assert str(create_embeddings().openai_api_base or "").find("orcarouter") == -1
    with pytest.raises(ValueError):
        create_embeddings(provider="typo")
    with pytest.raises(ValueError):
        create_embeddings(provider="orcarouter")


def test_no_insecure_remote_auth(monkeypatch):
    monkeypatch.setenv("ORCA_AUTH_BASE_URL", "http://example.com")
    with pytest.raises(ValueError):
        orcarouter.begin_login("http://127.0.0.1:51733/cb", "test-state")


def test_long_unicode_documents_and_queries(gateway, monkeypatch, tmp_path):
    monkeypatch.setenv("ORCAROUTER_API_KEY", "sk-orca-test")
    text = "apple 海洋🐋 " * 4000
    search = SemanticSearch(enable_bm25=False)
    search.create_vector_store([Document(page_content=text)], str(tmp_path / "long"))
    chunks = [chunk for _, _, data in gateway for chunk in data["input"]]
    assert len(chunks) > 1
    assert "".join(chunks) == text
    assert all(len(chunk.encode("utf-8")) <= 2000 for chunk in chunks)
    assert search.vector_store.index.ntotal == 1
    assert search.search(text, top_k=1)[0][0].page_content == text


def test_async_long_inputs_and_order(gateway, monkeypatch):
    import asyncio

    monkeypatch.setenv("ORCAROUTER_API_KEY", "sk-orca-test")
    embedding = create_embeddings()
    vectors = asyncio.run(embedding.aembed_documents(["apple " * 9000, "ocean"]))
    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    assert asyncio.run(embedding.aembed_query("apple " * 9000)) == [1.0, 0.0]


def test_chunk_vectors_use_weighted_normalized_mean(monkeypatch):
    from indexrag.providers.length_safe import LengthSafeEmbeddings

    class Backend:
        model = "test-model"
        chunk_size = 32

        def embed_documents(self, texts):
            return [[1.0, 0.0] if t[0] == "a" else [0.0, 1.0] for t in texts]

    e = LengthSafeEmbeddings(Backend())
    vector = e.embed_query("a" * 2000 + "b" * 1000)
    assert vector == pytest.approx([2 / (5**0.5), 1 / (5**0.5)])
    assert e.embed_documents([]) == []
    with pytest.raises(ValueError, match="empty"):
        e.embed_query("")


@pytest.mark.parametrize("action", ["logout", "new_login"])
def test_late_exchange_cannot_restore_or_overwrite_credentials(monkeypatch, action):
    orcarouter.save_credential("sk-orca-initial", "api")
    old_attempt = orcarouter.start_login()
    arrived, release = threading.Event(), threading.Event()
    errors = []

    def exchange(*args, **kwargs):
        arrived.set()
        assert release.wait(5)
        return httpx.Response(200, json={"key": "sk-orca-stale", "scope": "api"})

    def finish():
        try:
            orcarouter.exchange_code("old-code", "old-verifier", old_attempt)
        except ValueError as exc:
            errors.append(str(exc))

    monkeypatch.setattr(orcarouter.httpx, "post", exchange)
    thread = threading.Thread(target=finish)
    thread.start()
    assert arrived.wait(5)
    if action == "logout":
        orcarouter.logout()
    else:
        attempt = orcarouter.start_login()
        orcarouter.save_credential("sk-orca-new", "api", attempt)
    release.set()
    thread.join(5)
    assert not thread.is_alive()
    assert errors and "superseded" in errors[0]
    if action == "logout":
        assert not orcarouter.credential_path().exists()
    else:
        assert orcarouter.api_key() == "sk-orca-new"


def test_logout_in_another_process_invalidates_pending_login(monkeypatch):
    import subprocess
    import sys

    attempt = orcarouter.start_login()
    subprocess.run([sys.executable, "-m", "indexrag.providers.orcarouter", "logout"], check=True, capture_output=True)
    with pytest.raises(ValueError, match="superseded"):
        orcarouter.save_credential("sk-orca-late", "api", attempt)
    assert not orcarouter.credential_path().exists()


def test_non_unit_vectors_preserve_long_short_retrieval_ranking():
    import asyncio

    from langchain_community.vectorstores import FAISS

    from indexrag.providers.length_safe import LengthSafeEmbeddings

    class Backend:
        model = "custom-embedding"
        chunk_size = 32

        def embed_documents(self, texts):
            return [[10.0, 0.0] if text.startswith("a") else [8.0, 6.0] for text in texts]

        async def aembed_documents(self, texts):
            return self.embed_documents(texts)

    embedding = LengthSafeEmbeddings(Backend())
    documents = [
        Document(page_content="a" * 4000, metadata={"topic": "matching"}),
        Document(page_content="b", metadata={"topic": "unrelated"}),
    ]
    store = FAISS.from_documents(documents, embedding)
    hits = store.similarity_search_with_score("a", k=2)
    assert hits[0][0].metadata["topic"] == "matching"
    assert float(hits[0][1]) == pytest.approx(0.0)
    assert embedding.embed_query("a") == [1.0, 0.0]
    assert embedding.embed_query("b") == pytest.approx([0.8, 0.6])
    assert asyncio.run(embedding.aembed_query("a")) == [1.0, 0.0]
    assert asyncio.run(embedding.aembed_documents(["a", "a" * 4000])) == [[1.0, 0.0], [1.0, 0.0]]
