"""
Bridging facts generator — synthesize cross-document facts using LLM.
"""

import json
import logging
import threading
from typing import Dict, List, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .entity_linker import (
    find_bridge_entities,
    collect_entity_facts,
    build_source_faq_index,
)

logger = logging.getLogger("indexrag.bridging_facts")

BRIDGING_PROMPT = """Given the following information about "{entity}" from multiple source documents, generate bridging facts that connect information across these documents.

{doc_sections}

Requirements:
- Each bridging fact must combine information from 2+ documents
- Be factually accurate - only connect information that is logically related
- Each fact should be self-contained and understandable without context
- Do not generate speculative connections
- If documents share the entity name but are about unrelated topics, return empty

Return a JSON array of strings. If no meaningful connections exist, return [].
"""

MAX_DOCS_PER_ENTITY = 5
MAX_FACTS_PER_DOC = 8


def generate_bridging_facts_llm(
    entity: str,
    doc_facts_list: List[Tuple[str, List[str]]],
    model: str = "gpt-4o-mini",
) -> List[str]:
    """
    Call LLM to generate bridging facts for one entity across multiple documents.

    Args:
        entity: The bridge entity name.
        doc_facts_list: List of (source_name, facts_list) tuples.
        model: LLM model to use.

    Returns:
        List of bridging fact strings.
    """
    import openai

    sections = []
    for source_name, facts in doc_facts_list:
        facts_text = (
            "\n".join(facts[:MAX_FACTS_PER_DOC])
            if facts
            else "(No relevant facts found)"
        )
        sections.append(f"=== Document: {source_name} ===\n{facts_text}")

    prompt = BRIDGING_PROMPT.format(
        entity=entity,
        doc_sections="\n\n".join(sections),
    )

    client = openai.OpenAI()
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You generate bridging facts that connect information across documents. Return only a JSON array of strings.",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=2000,
        temperature=0.2,
    )

    response = completion.choices[0].message.content.strip()

    try:
        if response.startswith("```"):
            response = response.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        facts = json.loads(response)
        if isinstance(facts, list):
            return [f for f in facts if isinstance(f, str) and f.strip()]
        return []
    except json.JSONDecodeError:
        if response and response != "[]":
            return [response]
        return []


def generate_all_bridging_facts(
    faq_data: List[Dict[str, Any]],
    output_path: Optional[Path] = None,
    min_entity_len: int = 4,
    min_doc_freq: int = 2,
    max_doc_freq: int = 10,
    concurrency: int = 10,
    save_interval: int = 200,
    model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """
    Generate bridging facts for all cross-document bridge entities.

    Args:
        faq_data: List of FAQ extraction results.
        output_path: Optional path to save results (supports resume).
        min_entity_len: Minimum entity string length.
        min_doc_freq: Minimum document frequency for bridge entities.
        max_doc_freq: Maximum document frequency.
        concurrency: Number of concurrent LLM calls.
        save_interval: Save checkpoint every N entities.
        model: LLM model to use.

    Returns:
        Dict with 'config', 'stats', and 'bridging_facts' keys.
    """
    # Step 1: Build source-FAQ index
    source_faqs = build_source_faq_index(faq_data)

    # Step 2: Find bridge entities
    bridge_entities = find_bridge_entities(
        faq_data, min_entity_len, min_doc_freq, max_doc_freq
    )

    # Step 3: Load existing progress if output_path exists
    existing_results = []
    existing_keys = set()
    if output_path and output_path.exists():
        logger.info(f"Loading existing progress from {output_path}")
        with open(output_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        existing_results = existing.get("bridging_facts", [])
        existing_keys = {r["entity"] for r in existing_results}
        logger.info(f"Found {len(existing_keys)} completed entities")

    # Step 4: Prepare work items
    work_entities = []
    for entity, sources in sorted(bridge_entities.items()):
        if entity in existing_keys:
            continue

        sorted_sources = sorted(sources)
        if len(sorted_sources) > MAX_DOCS_PER_ENTITY:
            source_fact_counts = [
                (s, len(collect_entity_facts(source_faqs.get(s, []), entity)))
                for s in sorted_sources
            ]
            source_fact_counts.sort(key=lambda x: -x[1])
            sorted_sources = [s for s, _ in source_fact_counts[:MAX_DOCS_PER_ENTITY]]

        doc_facts_list = []
        for src in sorted_sources:
            facts = collect_entity_facts(source_faqs.get(src, []), entity)
            doc_facts_list.append((src, facts))

        work_entities.append((entity, doc_facts_list, sorted_sources))

    logger.info(
        f"Generating bridging facts: {len(work_entities)} remaining "
        f"(of {len(bridge_entities)} total entities)"
    )

    if not work_entities:
        logger.info("All entities already processed!")
        return _build_output(bridge_entities, existing_results, 0,
                             min_entity_len, min_doc_freq, max_doc_freq)

    # Step 5: Process with thread pool
    shutdown_event = threading.Event()
    results = list(existing_results)
    lock = threading.Lock()
    completed = 0
    errors = 0
    last_save = 0

    def _save_progress(incomplete=True):
        if not output_path:
            return
        data = _build_output(
            bridge_entities, results, errors,
            min_entity_len, min_doc_freq, max_doc_freq,
        )
        if incomplete:
            data["incomplete"] = True
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _process_entity(args):
        entity, doc_facts_list, sources = args
        if shutdown_event.is_set():
            return {"entity": entity, "skipped": True}
        try:
            bridging = generate_bridging_facts_llm(entity, doc_facts_list, model)
            return {
                "entity": entity,
                "sources": sources,
                "num_docs": len(doc_facts_list),
                "bridging_facts": bridging,
                "success": True,
            }
        except Exception as e:
            return {
                "entity": entity,
                "sources": sources,
                "num_docs": len(doc_facts_list),
                "bridging_facts": [],
                "success": False,
                "error": str(e),
            }

    try:
        batch_size = concurrency * 3
        total_work = len(work_entities)

        for batch_start in range(0, total_work, batch_size):
            if shutdown_event.is_set():
                break

            batch = work_entities[batch_start : batch_start + batch_size]

            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = {
                    executor.submit(_process_entity, item): item[0]
                    for item in batch
                }

                for future in as_completed(futures):
                    if shutdown_event.is_set():
                        break

                    result = future.result()
                    if result.get("skipped"):
                        continue

                    with lock:
                        results.append(result)
                        if not result.get("success"):
                            errors += 1
                        completed += 1

                        if completed % 100 == 0 or completed == total_work:
                            n_facts = sum(
                                len(r.get("bridging_facts", [])) for r in results
                            )
                            logger.info(
                                f"Progress: {completed}/{total_work} "
                                f"({100*completed//total_work}%) | "
                                f"Total facts: {n_facts} | Errors: {errors}"
                            )

                        if completed - last_save >= save_interval:
                            last_save = completed
                            _save_progress(incomplete=True)

    except KeyboardInterrupt:
        logger.info("Interrupted — saving progress...")
        shutdown_event.set()
        _save_progress(incomplete=True)
        logger.info(f"Saved {len(results)} entities. Run again to resume.")

    # Final save
    output_data = _build_output(
        bridge_entities, results, errors,
        min_entity_len, min_doc_freq, max_doc_freq,
    )
    if output_path:
        _save_progress(incomplete=False)
        logger.info(f"Saved to {output_path}")

    return output_data


def _build_output(
    bridge_entities, results, errors,
    min_entity_len, min_doc_freq, max_doc_freq,
) -> Dict[str, Any]:
    total_facts = sum(len(r.get("bridging_facts", [])) for r in results)
    non_empty = sum(1 for r in results if r.get("bridging_facts"))
    return {
        "config": {
            "min_entity_len": min_entity_len,
            "max_doc_freq": max_doc_freq,
            "min_doc_freq": min_doc_freq,
            "max_docs_per_entity": MAX_DOCS_PER_ENTITY,
            "max_facts_per_doc": MAX_FACTS_PER_DOC,
        },
        "stats": {
            "total_entities": len(bridge_entities),
            "completed_entities": len(results),
            "total_bridging_facts": total_facts,
            "non_empty_entities": non_empty,
            "errors": errors,
        },
        "bridging_facts": results,
    }
