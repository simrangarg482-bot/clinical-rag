"""
LangGraph orchestration — wires every module built so far into one
stateful agent graph with an explicit Corrective RAG retry loop.

Graph shape:

  input_guard --> retrieve --> self_rag --(insufficient_context)--> corrective_rag --> retrieve (loop)
                                   |
                                (success)
                                   v
                              output_guard --> END

  - State is a typed TypedDict, not a loose dict — every field's shape
    is explicit and IDE-checkable.
  - The corrective loop is BOUNDED by settings.max_self_rag_retries.
    Without this cap, a query that's genuinely unanswerable from the
    corpus would loop forever, burning LLM calls. After the cap, we
    proceed to generation anyway with whatever was found, rather than
    silently failing — see route_after_self_rag.
  - Guardrail short-circuits are explicit graph edges to END, not
    exceptions — a blocked/withheld request is a normal, loggable
    outcome of the graph, not an error condition.
  - Every node logs its own decision, so a full request can be traced
    end-to-end from the logs alone (useful for both debugging and for
    a resume demo showing "look, here's the full reasoning trace").
"""
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END
from loguru import logger

from config import settings
from retrieval.hybrid_retrieval import hybrid_retrieve, RetrievedDoc
from retrieval.reranker import rerank
from rag_pipeline.self_rag import self_rag_pipeline, rewrite_query
from rag_guardrails.input_guard import check_input
from rag_guardrails.output_guard import check_output
from retrieval.web_search import web_search_available, run_web_search, generate_from_web


class ClinicalState(TypedDict):
    query: str                          # original user query, never mutated
    active_query: str                   # current query used for retrieval (may be rewritten)
    candidate_docs: list[RetrievedDoc]   # post-retrieval, post-rerank
    self_rag_result: Optional[dict]      # output of self_rag_pipeline()
    retries: int                        # corrective RAG retry counter
    final_answer: Optional[str]          # what gets shown to the user
    blocked_reason: Optional[str]  
    web_search_enabled: bool        # ← new: user opt-in toggle
    source_type: str       # set if input/output guard rejected


# --- Nodes ---

def input_guard_node(state: ClinicalState) -> ClinicalState:
    safe, reason = check_input(state["query"])
    if not safe:
        logger.warning(f"Input blocked: {reason}")
        return {**state, "blocked_reason": reason}
    return state


def retrieve_node(state: ClinicalState) -> ClinicalState:
    query = state["active_query"]
    raw_docs = hybrid_retrieve(query, top_k=settings.retrieval_top_k)
    top_docs = rerank(query, raw_docs, top_k=settings.rerank_top_k)
    logger.info(f"Retrieved + reranked {len(top_docs)} candidate docs for: '{query}'")
    return {**state, "candidate_docs": top_docs}


def self_rag_node(state: ClinicalState) -> ClinicalState:
    result = self_rag_pipeline(state["query"], state["candidate_docs"])
    return {**state, "self_rag_result": result}


def corrective_rag_node(state: ClinicalState) -> ClinicalState:
    new_query = rewrite_query(state["active_query"])
    return {
        **state,
        "active_query": new_query,
        "retries": state["retries"] + 1,
    }

def web_search_node(state: ClinicalState) -> ClinicalState:
    if not web_search_available():
        logger.warning("Web search triggered but TAVILY_API_KEY not set — skipping.")
        return {**state, "self_rag_result": {"status": "insufficient_context", "relevant_docs": []}}
    try:
        results = run_web_search(state["query"])
        result = generate_from_web(state["query"], results)
        return {**state, "self_rag_result": result, "source_type": "web"}
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return {**state, "self_rag_result": {"status": "insufficient_context", "relevant_docs": []}}
    

def route_after_self_rag(state: ClinicalState) -> str:
    result = state["self_rag_result"]
    if result["status"] == "success":
        return "output_guard"
    if state["retries"] < settings.max_self_rag_retries:
        logger.info(
            f"Insufficient context (retry {state['retries'] + 1}/"
            f"{settings.max_self_rag_retries}) — triggering Corrective RAG."
        )
        return "corrective_rag"
    # Retries exhausted — try web search if enabled, otherwise honest non-answer
    if state.get("web_search_enabled") and web_search_available():
        logger.info("Corpus retries exhausted — falling back to web search.")
        return "web_search"
    logger.warning(f"Corrective RAG retries exhausted ({state['retries']}). Proceeding with honest non-answer.")
    return "output_guard"


def output_guard_node(state: ClinicalState) -> ClinicalState:
    result = state["self_rag_result"]

    if result["status"] != "success":
        # Exhausted retries without enough relevant context — be honest
        # about it rather than generating an unsupported answer.
        final = (
            "I wasn't able to find sufficient relevant information in the "
            "clinical guideline corpus to answer this question confidently. "
            "Please consult current clinical guidelines or a licensed clinician directly."
        )
        return {**state, "final_answer": final}

    answer = result["answer"]
    safe, reason = check_output(answer)
    if not safe:
        logger.warning(f"Output withheld: {reason}")
        return {**state, "blocked_reason": reason, "final_answer": None}

    disclaimer = (
        "\n\n⚠️ This is an AI-generated clinical summary based on the ingested "
        "guideline corpus. Always verify against current guidelines and "
        "exercise independent clinical judgment."
    )
    return {**state, "final_answer": answer + disclaimer}


# --- Conditional routing ---

def route_after_input_guard(state: ClinicalState) -> str:
    return END if state.get("blocked_reason") else "retrieve"


def route_after_self_rag(state: ClinicalState) -> str:
    result = state["self_rag_result"]
    if result["status"] == "success":
        return "output_guard"
    if state["retries"] < settings.max_self_rag_retries:
        logger.info(
            f"Insufficient context (retry {state['retries'] + 1}/"
            f"{settings.max_self_rag_retries}) — triggering Corrective RAG."
        )
        return "corrective_rag"
    # Retries exhausted — proceed to output_guard anyway, which will
    # produce the honest "couldn't find enough information" message
    # via the status != "success" branch above, rather than looping forever.
    logger.warning(f"Corrective RAG retries exhausted ({state['retries']}). Proceeding with honest non-answer.")
    return "output_guard"


# --- Build the graph ---

def build_graph():
    graph = StateGraph(ClinicalState)

    graph.add_node("input_guard", input_guard_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("self_rag", self_rag_node)
    graph.add_node("corrective_rag", corrective_rag_node)
    graph.add_node("output_guard", output_guard_node)
    graph.add_node("web_search", web_search_node)

    graph.set_entry_point("input_guard")
    graph.add_conditional_edges(
        "input_guard", route_after_input_guard, {"retrieve": "retrieve", END: END}
    )
    graph.add_edge("retrieve", "self_rag")
    graph.add_conditional_edges(
        "self_rag",
        route_after_self_rag,
        {
        "corrective_rag": "corrective_rag",
        "output_guard": "output_guard",
        "web_search": "web_search",
        },
    )
    graph.add_edge("corrective_rag", "retrieve")
    graph.add_edge("web_search", "output_guard")
    graph.add_edge("output_guard", END)

    return graph.compile()


clinical_rag_app = build_graph()


def ask(query: str, web_search_enabled: bool = False) -> dict:
    initial_state: ClinicalState = {
        "query": query,
        "active_query": query,
        "candidate_docs": [],
        "self_rag_result": None,
        "retries": 0,
        "final_answer": None,
        "blocked_reason": None,
        "web_search_enabled": web_search_enabled,
        "source_type": "corpus",
    }
    logger.info(f"=== New query: '{query}' (web_search={'on' if web_search_enabled else 'off'}) ===")
    final_state = clinical_rag_app.invoke(initial_state)
    logger.info("=== Query complete ===")
    return final_state