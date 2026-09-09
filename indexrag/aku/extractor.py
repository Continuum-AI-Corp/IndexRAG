"""
AKU Extractor — extract Atomic Knowledge Units from text chunks using LLM.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

from .schema import AKU, AKUExtractionResult
from .prompts.loader import load_system_prompt, load_user_prompt

logger = logging.getLogger("indexrag.aku")


def _parse_json_response(response: str) -> Optional[Dict]:
    """Parse JSON from LLM response, handling markdown code blocks."""
    if not response:
        return None

    # Strip markdown code blocks
    if "```json" in response:
        json_str = response.split("```json")[1].split("```")[0].strip()
    elif "```" in response:
        json_str = response.split("```")[1].split("```")[0].strip()
    else:
        json_str = response.strip()

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


def _clean_aku(faq: Dict) -> Optional[AKU]:
    """Validate and clean a single FAQ dict into an AKU."""
    if not isinstance(faq, dict):
        return None
    if not faq.get("question") or not faq.get("answer"):
        return None

    entities = faq.get("entities", [])
    if isinstance(entities, list):
        entities = [e for e in entities if isinstance(e, str) and e.strip()]
    else:
        entities = []

    return AKU(
        question=faq["question"],
        answer=faq["answer"],
        entities=entities,
    )


def extract_akus_from_chunk(
    chunk: str,
    chunk_id: str = "",
    source: str = "",
    model: str = "gpt-4o-mini",
    max_tokens: int = 8000,
    custom_prompt_file: Optional[str] = None,
) -> AKUExtractionResult:
    """
    Extract AKUs from a text chunk using LLM.

    Args:
        chunk: Text content to extract AKUs from.
        chunk_id: Identifier for this chunk.
        source: Source document name.
        model: LLM model to use.
        max_tokens: Maximum output tokens.
        custom_prompt_file: Path to custom system prompt (with {text} placeholder).

    Returns:
        AKUExtractionResult with extracted AKUs.
    """
    import openai

    # Build prompt
    if custom_prompt_file:
        with open(Path(custom_prompt_file), "r", encoding="utf-8") as f:
            sys_prompt = f.read().replace("{text}", chunk)
        user_prompt = "Extract all facts as specified above."
    else:
        sys_prompt = load_system_prompt().replace("{text}", chunk)
        user_prompt = load_user_prompt()

    try:
        client = openai.OpenAI()
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        response = completion.choices[0].message.content

        parsed = _parse_json_response(response)
        if not parsed or "all_faqs" not in parsed:
            return AKUExtractionResult(
                chunk_id=chunk_id, source=source, error="Failed to parse JSON response"
            )

        akus = []
        for faq in parsed["all_faqs"]:
            aku = _clean_aku(faq)
            if aku:
                akus.append(aku)

        return AKUExtractionResult(
            chunk_id=chunk_id,
            source=source,
            akus=akus,
            metadata={
                "model": model,
                "input_chars": len(chunk),
                "num_akus": len(akus),
            },
        )

    except Exception as e:
        logger.warning(f"AKU extraction failed for chunk {chunk_id}: {e}")
        return AKUExtractionResult(
            chunk_id=chunk_id, source=source, error=str(e)
        )


def extract_summary_from_chunk(
    chunk: str,
    custom_prompt_file: str,
    model: str = "gpt-4o-mini",
    max_tokens: int = 8000,
) -> str:
    """
    Generate a free-text summary from a text chunk using LLM.

    Args:
        chunk: Text content to summarize.
        custom_prompt_file: Path to prompt file with {text} placeholder (required).
        model: LLM model to use.
        max_tokens: Maximum output tokens.

    Returns:
        Raw LLM response text.
    """
    import openai

    with open(Path(custom_prompt_file), "r", encoding="utf-8") as f:
        sys_prompt = f.read().replace("{text}", chunk)

    try:
        client = openai.OpenAI()
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": "Process the document as specified above."},
            ],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        response = completion.choices[0].message.content
        return response.strip() if response else ""
    except Exception as e:
        logger.error(f"Summary extraction failed: {e}")
        return ""
