"""
Web search fallback for out-of-scope queries.

Only triggered when the corpus pipeline returns "insufficient_context"
after all Corrective RAG retries are exhausted AND the user has
explicitly enabled web search via the UI toggle.

Design decisions:
  - Tavily is used instead of raw Google/Bing because it returns
    clean, pre-extracted text snippets rather than raw HTML — much
    more reliable for direct use as RAG context.
  - Web results are explicitly labeled as unvetted in the returned
    dict so the UI can render them with a distinct visual treatment
    (amber/warning card) vs. corpus-grounded answers (green card).
  - The same CLINICAL_SYSTEM_PROMPT and generation prompt are used
    so the answer style stays consistent — but the source provenance
    label changes.
  - Web search is DISABLED by default and requires explicit opt-in
    from the user. In a clinical context, the difference between a
    vetted guideline and a random medical website matters.
"""
import os

from loguru import logger
from tavily import TavilyClient

from config import call_llm, settings, CLINICAL_SYSTEM_PROMPT

WEB_GENERATION_PROMPT = """You are a clinical information assistant. A user asked a medical question
that was not covered by the ingested guideline corpus, so web search results are being used instead.

Answer the query using ONLY the information in the web search results below.
Cite each claim with [Web N], referring to the result numbers given.

IMPORTANT: These are web search results, NOT vetted clinical guidelines.
If the results are unclear or contradictory, say so explicitly rather than guessing.
Always recommend the user verify with current official guidelines and a licensed clinician.

Query: {query}

Web search results:
{results}

Answer:"""


def web_search_available() -> bool:
    """Returns True only if a Tavily key is configured."""
    return bool(settings.tavily_api_key)


def run_web_search(query: str, max_results: int = 4) -> list[dict]:
    """
    Runs a Tavily search and returns a list of result dicts:
    [{"title": str, "url": str, "content": str}, ...]
    """
    if not web_search_available():
        raise RuntimeError(
            "TAVILY_API_KEY is not set. Add it to .env to enable web search fallback."
        )
    client = TavilyClient(api_key=settings.tavily_api_key)
    # Append "clinical guidelines" to bias results toward medical reference sources
    search_query = f"{query} clinical guidelines"
    logger.info(f"Web search fallback: '{search_query}'")
    response = client.search(query=search_query, max_results=max_results, search_depth="advanced")
    results = response.get("results", [])
    logger.info(f"Web search returned {len(results)} results.")
    return results


def generate_from_web(query: str, results: list[dict]) -> dict:
    """
    Generates a grounded answer from web search results and returns
    a dict shaped similarly to self_rag_pipeline's success output,
    but with source_type="web" so the graph and UI can distinguish it.
    """
    results_text = "\n\n".join(
        f"[Web {i + 1}] {r.get('title', 'Untitled')} ({r.get('url', '')})\n{r.get('content', '')}"
        for i, r in enumerate(results)
    )
    prompt = WEB_GENERATION_PROMPT.format(query=query, results=results_text)
    answer = call_llm(prompt, temperature=0.1, system_prompt=CLINICAL_SYSTEM_PROMPT).strip()

    return {
        "status": "success",
        "source_type": "web",    # ← key flag for UI differentiation
        "answer": answer,
        "web_sources": results,
        "relevant_docs": [],     # no corpus docs used
        "support": {"verdict": "web_sourced", "explanation": "Answer generated from web search results, not vetted clinical guidelines."},
    }