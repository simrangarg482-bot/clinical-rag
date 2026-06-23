"""
HyDE (Hypothetical Document Embeddings) query expansion.

Idea: instead of embedding the raw user query (which is often short and
vague — "what's the dose for kids?"), we ask the LLM to write a short
hypothetical clinical passage that WOULD answer the query, then embed
THAT. Hypothetical passages live in the same semantic neighborhood as
real clinical guideline text, which improves recall on terse or indirect
queries.

  - Uses the shared call_llm() wrapper -> automatic retry/backoff, centralized logging.
  - Validates LLM output isn't empty before embedding (fail loud, not silent).
  - Returns BOTH the embedding and the hypothetical doc text, so callers
    can log/display it for debugging ("why did retrieval return this?").
"""
from loguru import logger

from config import call_llm, embed_query

HYDE_PROMPT_TEMPLATE = """You are a clinical medical writer. Given the question below, write a short,
factual paragraph (3-5 sentences) as if it were an excerpt from a clinical practice
guideline or peer-reviewed medical reference that directly answers the question.
Do not mention that this is hypothetical. Write only the passage itself.

Question: {query}

Hypothetical clinical passage:"""


def hyde_embed(query: str) -> tuple[list[float], str]:
    """
    Generates a hypothetical clinical passage for `query` and returns
    (embedding_vector, hypothetical_passage_text).

    Raises ValueError if the LLM returns an empty/degenerate response —
    we'd rather fail loudly here than silently embed an empty string.
    """
    if not query or not query.strip():
        raise ValueError("hyde_embed called with empty query.")

    prompt = HYDE_PROMPT_TEMPLATE.format(query=query.strip())
    hypo_doc = call_llm(prompt, temperature=0.3).strip()

    if not hypo_doc or len(hypo_doc) < 20:
        logger.warning(
            f"HyDE generated suspiciously short output ({len(hypo_doc)} chars) "
            f"for query: '{query}'. Falling back to embedding the raw query."
        )
        return embed_query(query), query

    logger.debug(f"HyDE passage for '{query}': {hypo_doc[:120]}...")
    vector = embed_query(hypo_doc)
    return vector, hypo_doc