"""OrcaRouter PKCE login and user-local credential storage (no client secret)."""

import argparse
import base64
import getpass
import hashlib
import json
import os
import secrets
import tempfile
import time
import webbrowser
from pathlib import Path
from urllib.parse import urlencode, urlsplit

import httpx


def origin(kind):
    default = "https://www.orcarouter.ai" if kind == "AUTH" else "https://api.orcarouter.ai"
    value = os.getenv("ORCA_" + kind + "_BASE_URL") or os.getenv("ORCA_BASE_URL") or default
    value = value.rstrip("/")
    parsed = urlsplit(value)
    if (
        parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or not parsed.hostname
        or parsed.scheme not in ("https", "http")
        or (parsed.scheme == "http" and parsed.hostname not in ("localhost", "127.0.0.1", "::1"))
    ):
        raise ValueError("OrcaRouter URLs must use HTTPS (HTTP is allowed only on loopback).")
    return value


def api_url():
    value = origin("API")
    return value if value.endswith("/v1") else value + "/v1"


def credential_path():
    root = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
    return root / "indexrag" / "orcarouter.json"


def read_credential():
    path = credential_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise TypeError()
        return data
    except (ValueError, TypeError, OSError):
        raise ValueError("Cannot read OrcaRouter credentials; run indexrag-auth login again.") from None


def api_key():
    explicit = os.getenv("ORCAROUTER_API_KEY")
    if explicit:
        return explicit
    saved = read_credential()
    if saved.get("api_base_url") != api_url():
        raise ValueError("No OrcaRouter login for this API URL. Set ORCAROUTER_API_KEY or run indexrag-auth login.")
    key = saved.get("key")
    if not isinstance(key, str) or not key:
        raise ValueError("Missing OrcaRouter API key. Set ORCAROUTER_API_KEY or run indexrag-auth login.")
    return key


def save_credential(key, scope):
    path = credential_path()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    fd, temporary = tempfile.mkstemp(prefix=".orcarouter-", dir=path.parent)
    try:
        os.chmod(temporary, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"key": key, "scope": scope, "api_base_url": api_url()}, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def begin_login():
    verifier = secrets.token_urlsafe(48)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest()).rstrip(b"=").decode("ascii")
    url = (
        origin("AUTH")
        + "/auth?"
        + urlencode(
            {
                "callback_url": "oob",
                "code_challenge": challenge,
                "code_challenge_method": "S256",
                "state": secrets.token_urlsafe(32),
                "app_name": "IndexRAG",
                "scope": "api",
            }
        )
    )
    return verifier, url


def exchange_code(code, verifier):
    if not code.strip():
        raise ValueError("Authorization code is empty.")
    try:
        response = httpx.post(
            origin("AUTH") + "/api/v1/auth/keys",
            json={
                "code": code.strip(),
                "code_verifier": verifier,
                "code_challenge_method": "S256",
            },
            timeout=30,
            follow_redirects=False,
        )
        if response.status_code != 200:
            raise ValueError(f"OrcaRouter authorization failed (HTTP {response.status_code}); start a new login.")
        data = response.json()
    except httpx.HTTPError:
        raise ValueError("Cannot reach OrcaRouter authorization server; try again.") from None
    except json.JSONDecodeError:
        raise ValueError("Invalid OrcaRouter authorization response.") from None
    if not isinstance(data, dict) or not isinstance(data.get("key"), str) or not data["key"].startswith("sk-orca-"):
        raise ValueError("OrcaRouter authorization did not return a valid API key.")
    if data.get("scope") != "api":
        raise ValueError("OrcaRouter did not grant the requested api scope; credentials were not saved.")
    save_credential(data["key"], data["scope"])


def main():
    parser = argparse.ArgumentParser(description="OrcaRouter embedding credentials for IndexRAG")
    parser.add_argument("command", choices=("login", "status", "logout"))
    parser.add_argument(
        "--no-browser", action="store_true", help="Print the authorization URL without opening a browser"
    )
    args = parser.parse_args()
    try:
        if args.command == "login":
            verifier, url = begin_login()
            started = time.monotonic()
            print("Open this URL, approve IndexRAG, then paste the displayed authorization code:\n" + url)
            if not args.no_browser:
                webbrowser.open(url)
            code = getpass.getpass("Authorization code: ")
            if time.monotonic() - started >= 600:
                raise ValueError("Login expired; run indexrag-auth login again.")
            exchange_code(code, verifier)
            print("OrcaRouter login saved. Embeddings can use INDEXRAG_EMBEDDING_PROVIDER=orcarouter.")
        elif args.command == "logout":
            credential_path().unlink(missing_ok=True)
            print(
                "Local OrcaRouter login removed. Environment keys are unchanged; revoke keys in your OrcaRouter console."
            )
        else:
            api_key()
            print("OrcaRouter embedding credentials configured (not a connectivity check).")
    except (ValueError, OSError) as exc:
        parser.exit(1, str(exc) + "\n")
    except (KeyboardInterrupt, EOFError):
        parser.exit(1, "\nLogin canceled; previous credentials are unchanged.\n")


if __name__ == "__main__":
    main()
