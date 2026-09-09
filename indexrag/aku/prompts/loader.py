"""Load AKU extraction prompts."""

from pathlib import Path

_PROMPT_DIR = Path(__file__).parent


def load_system_prompt(name: str = "sys_extract.md") -> str:
    """Load system prompt template."""
    with open(_PROMPT_DIR / name, "r", encoding="utf-8") as f:
        return f.read()


def load_user_prompt(name: str = "user_extract.md") -> str:
    """Load user prompt template."""
    with open(_PROMPT_DIR / name, "r", encoding="utf-8") as f:
        return f.read()
