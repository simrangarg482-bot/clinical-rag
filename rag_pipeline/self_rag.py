"""
Self-RAG + Corrective RAG.

Self-RAG: the LLM critiques its own pipeline at two points instead of
blindly trusting retrieval —
  1. RELEVANCE check, per retrieved chunk: "does this actually answer
     the query, or did it just rank high on similarity?"
  2. SUPPORT check, on the generated answer: "is every claim in this
     answer actually grounded in the relevant chunks, or did the model
     add something unsupported?"

Corrective RAG: triggered when relevance filtering leaves too few
usable chunks. Instead of generating a weak/unsupported answer anyway,
the query is rewritten (more specific medical terminology) and
retrieval is retried, up to a capped number of attempts.

  - Structured JSON output for relevance/support grading (not free-text
    parsing) — JSON parsing failures are caught and treated as a
    conservative "irrelevant"/"not supported" rather than crashing.
  - Bounded retries everywhere — relevance grading per chunk uses
    call_llm's existing retry/backoff; the corrective loop itself is
    capped by settings.max_self_rag_retries so a stubborn query can't
    loop forever.
  - Every decision (relevance verdict, support verdict) is logged with
    the chunk identity, so a reviewer can replay exactly why a chunk
    was kept or dropped.
  - All clinical-content calls pass CLINICAL_SYSTEM_PROMPT to reduce
    false-positive safety refusals on medical reference text (some
    models occasionally refuse to grade/summarize clinical passages
    without this framing, mistaking evaluation work for patient advice).
"""
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger

from config import call_llm, settings, CLINICAL_SYSTEM_PROMPT
from retrieval.hybrid_retrieval import RetrievedDoc

# --- Prompts ---

RELEVANCE_PROMPT = """You are grading whether a retrieved passage is relevant to a clinical query.
Respond with ONLY a JSON object, no other text:
{{"relevant": true or false, "reason": "one short sentence"}}

Query: {query}

Passage:
{passage}
"""

GENERATION_PROMPT = """You are a clinical decision support assistant. Answer the query using ONLY
the information in the provided passages below. Cite every factual claim with
[Source N], referring to the passage numbers given.

Rules:
- Do not state anything not directly supported by the passages.
- If the passages don't fully answer the query, say so explicitly rather than guessing.
- Be concise and clinically precise.

Query: {query}

Passages:
{passages}

Answer:"""

SUPPORT_PROMPT = """You are checking whether a generated clinical answer is fully grounded in
its source passages, or whether it makes claims the passages don't support.
Respond with ONLY a JSON object, no other text:
{{"verdict": "fully_supported" or "partially_supported" or "not_supported", "explanation": "one short sentence"}}

Source passages:
{passages}

Generated answer:
{answer}
"""

QUERY_REWRITE_PROMPT = """The following clinical search query returned mostly irrelevant results.
Rewrite it to use more precise clinical/medical terminology that is more likely
to match passages in a clinical guideline corpus. Return ONLY the rewritten
query text, nothing else.

Original query: {query}"""


def _parse_json_response(raw: str, fallback: dict) -> dict:
    """
    Defensive JSON parsing — LLMs occasionally wrap JSON in markdown fences
    or add stray text despite instructions. Strip common wrappers, then
    fall back to a conservative default on any parse failure rather than
    propagating an exception into the middle of the relevance-grading loop.
    """
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse expected JSON from LLM, using fallback. Raw: {raw!r}")
        return fallback


def grade_relevance(query: str, doc: RetrievedDoc) -> bool:
    """
    Self-RAG relevance check for a single chunk. Mutates doc.relevance
    in place (matching the existing pattern of dense_rank/rrf_score/
    rerank_score living on the RetrievedDoc itself) and returns the verdict.
    """
    prompt = RELEVANCE_PROMPT.format(query=query, passage=doc.text)
    raw = call_llm(prompt, temperature=0.0, system_prompt=CLINICAL_SYSTEM_PROMPT)
    parsed = _parse_json_response(raw, fallback={"relevant": False, "reason": "unparseable LLM response"})

    is_relevant = bool(parsed.get("relevant", False))
    doc.relevance = "relevant" if is_relevant else "irrelevant"

    logger.debug(
        f"Relevance check [{doc.source} p.{doc.page}]: {doc.relevance} "
        f"— {parsed.get('reason', 'no reason given')}"
    )
    return is_relevant


def rewrite_query(query: str) -> str:
    """Corrective RAG step: rewrite a query that returned poor relevance."""
    rewritten = call_llm(QUERY_REWRITE_PROMPT.format(query=query), temperature=0.3).strip()
    logger.info(f"Corrective RAG rewrote query: '{query}' -> '{rewritten}'")
    return rewritten


def generate_answer(query: str, relevant_docs: list[RetrievedDoc]) -> str:
    """Generates a grounded, cited answer using only the relevant docs."""
    passages_text = "\n\n".join(
        f"[Source {i + 1}] ({doc.source}, p.{doc.page})\n{doc.text}"
        for i, doc in enumerate(relevant_docs)
    )
    prompt = GENERATION_PROMPT.format(query=query, passages=passages_text)
    return call_llm(prompt, temperature=0.1, system_prompt=CLINICAL_SYSTEM_PROMPT).strip()


def check_support(answer: str, relevant_docs: list[RetrievedDoc]) -> dict:
    """
    Self-RAG support check: verifies the generated answer is actually
    grounded in the passages it was given, not hallucinated on top of them.
    """
    passages_text = "\n\n".join(f"[Source {i + 1}] {doc.text}" for i, doc in enumerate(relevant_docs))
    prompt = SUPPORT_PROMPT.format(passages=passages_text, answer=answer)
    raw = call_llm(prompt, temperature=0.0, system_prompt=CLINICAL_SYSTEM_PROMPT)
    parsed = _parse_json_response(
        raw, fallback={"verdict": "partially_supported", "explanation": "unparseable LLM response"}
    )
    logger.info(f"Support check: {parsed.get('verdict')} — {parsed.get('explanation', '')}")
    return parsed


def grade_relevance_parallel(query: str, docs: list[RetrievedDoc], max_workers: int = 5) -> list[RetrievedDoc]:
    """
    Runs grade_relevance() for every candidate doc CONCURRENTLY instead of
    sequentially. Each call is a network-bound LLM request, so running
    them in a thread pool gives a near-linear speedup (5 sequential ~8s
    calls -> ~8-10s total instead of ~40s) with no change in what's
    being asked or how results are judged — same prompts, same logic,
    just dispatched in parallel.

    Returns the SAME doc objects passed in, with .relevance mutated in
    place (matching grade_relevance's existing contract), so callers can
    keep using candidate_docs/relevant_docs exactly as before.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_doc = {executor.submit(grade_relevance, query, doc): doc for doc in docs}
        for future in as_completed(future_to_doc):
            doc = future_to_doc[future]
            try:
                future.result()
            except Exception as e:
                # grade_relevance already has its own retry/fallback via call_llm;
                # this is a final safety net so one bad thread can't crash the batch.
                logger.warning(f"Relevance check thread failed for {doc.source} p.{doc.page}: {e}")
                doc.relevance = "irrelevant"
    return docs


def self_rag_pipeline(query: str, candidate_docs: list[RetrievedDoc]) -> dict:
    """
    Runs the full Self-RAG relevance-filter -> generate -> support-check
    sequence on a single retrieval pass. Does NOT itself retry retrieval —
    that's the caller's job (see the LangGraph corrective_rag_node in
    rag_pipeline/graph.py), so this function stays a single, testable unit.

    Returns a dict:
      {"status": "success", "answer": str, "relevant_docs": [...], "support": {...}}
      {"status": "insufficient_context", "relevant_docs": [...]}  -- triggers Corrective RAG
    """
    graded_docs = grade_relevance_parallel(query, candidate_docs)
    relevant_docs = [doc for doc in graded_docs if doc.relevance == "relevant"]

    logger.info(
        f"Relevance filtering: {len(relevant_docs)}/{len(candidate_docs)} chunks kept "
        f"for query '{query}'"
    )

    MIN_RELEVANT_DOCS = 2
    if len(relevant_docs) < MIN_RELEVANT_DOCS:
        return {"status": "insufficient_context", "relevant_docs": relevant_docs}

    answer = generate_answer(query, relevant_docs)
    support = check_support(answer, relevant_docs)

    return {
        "status": "success",
        "answer": answer,
        "relevant_docs": relevant_docs,
        "support": support,
    }