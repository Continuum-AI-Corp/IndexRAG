"""OrcaRouter PKCE login and user-local credential storage (no client secret)."""

import argparse
import base64
import hashlib
import json
import os
import secrets
import tempfile
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit

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


class CallbackReceiver(HTTPServer):
    """One local authorization attempt. Unrelated requests cannot finish it."""

    def __init__(self):
        self.state = secrets.token_urlsafe(32)
        self.code = None
        self.denied = False
        self.failure = None
        self.on_code = None

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass  # Never log callback URLs, which contain authorization codes.

            def do_GET(self):
                parsed = urlsplit(self.path)
                params = parse_qs(parsed.query)
                valid = (
                    parsed.path == "/cb"
                    and len(self.path) <= 8192
                    and self.headers.get("Host") == self.server.authority
                    and params.get("state") is not None
                    and len(params["state"]) == 1
                    and secrets.compare_digest(params["state"][0], self.server.state)
                )
                content_type = "text/plain; charset=utf-8"
                if not valid:
                    status, message = 400, "Invalid login callback."
                elif len(params.get("error", [])) == 1 and "code" not in params:
                    self.server.denied = True
                    status, message = 200, "Authorization declined. You can close this tab."
                elif len(params.get("code", [])) == 1 and "error" not in params:
                    code = params["code"][0]
                    try:
                        if self.server.on_code:
                            self.server.on_code(code)
                    except (ValueError, OSError) as exc:
                        self.server.failure = exc
                        status, message = 400, "Login failed. Return to the terminal for details and try again."
                    else:
                        self.server.code = code
                        status = 200
                        content_type = "text/html; charset=utf-8"
                        message = """<!doctype html>
<html lang="en"><meta charset="utf-8"><title>IndexRAG login complete</title>
<body><h1>Login successful</h1>
<p>Your OrcaRouter credentials have been saved. This tab will close automatically.</p>
<p>If it stays open, your browser has blocked automatic closing. You can close it now.</p>
<script>
history.replaceState(null, "", "/cb");
setTimeout(function () { window.close(); }, 200);
</script></body></html>"""
                else:
                    status, message = 400, "Missing or ambiguous authorization code."
                body = message.encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Referrer-Policy", "no-referrer")
                self.end_headers()
                self.wfile.write(body)

        super().__init__(("127.0.0.1", 0), Handler)
        self.authority = f"127.0.0.1:{self.server_port}"
        self.callback_url = f"http://{self.authority}/cb"
        self.timeout = 0.5

    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(2)
        return connection, address

    def handle_error(self, request, client_address):
        pass  # A disconnected browser must not leak the callback in a traceback.

    def wait(self, seconds=600, on_code=None):
        self.on_code = on_code
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.handle_request()
            if self.failure:
                raise self.failure
            if self.denied:
                raise ValueError("OrcaRouter authorization was declined.")
            if self.code:
                return self.code
        raise ValueError("Login expired; run indexrag-auth login again.")


def begin_login(callback_url, state):
    verifier = secrets.token_urlsafe(48)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest()).rstrip(b"=").decode("ascii")
    url = (
        origin("AUTH")
        + "/auth?"
        + urlencode(
            {
                "callback_url": callback_url,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
                "state": state,
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
            with CallbackReceiver() as receiver:
                verifier, url = begin_login(receiver.callback_url, receiver.state)
                print("Open this URL and approve IndexRAG. The browser will return automatically:\n" + url, flush=True)
                if not args.no_browser:
                    webbrowser.open(url)
                receiver.wait(600, on_code=lambda code: exchange_code(code, verifier))
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
