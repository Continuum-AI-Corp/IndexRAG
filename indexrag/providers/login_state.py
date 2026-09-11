"""Cross-process ordering for login completion and logout."""

import json
import os
import secrets
import tempfile
from contextlib import contextmanager

from filelock import FileLock


@contextmanager
def locked_directory(path):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    with FileLock(str(path.parent / "orcarouter.lock"), timeout=10):
        yield


def generation(path):
    state = path.parent / "orcarouter-generation.json"
    if not state.exists():
        return None
    return json.loads(state.read_text(encoding="utf-8"))["generation"]


def invalidate(path):
    """Caller holds the directory lock. Persist before any credential mutation."""
    value = secrets.token_urlsafe(32)
    fd, name = tempfile.mkstemp(prefix=".generation-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"generation": value}, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path.parent / "orcarouter-generation.json")
    finally:
        if os.path.exists(name):
            os.unlink(name)
    return value
