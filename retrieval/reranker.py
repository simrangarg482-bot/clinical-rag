"""
Cross-encoder re-ranking.

Why this step exists, given we already have RRF-fused dense+sparse scores:
  Dense (embedding cosine similarity) and sparse (BM25) scores are both
  computed INDEPENDENTLY of each other per document — neither ever lets
  the query and passage attend to each other jointly. A cross-encoder
  does: it feeds (query, passage) through a single transformer together,
  so the model can directly judge "does THIS passage answer THIS query"
  rather than "is this passage's vector close to this query's vector".
  This consistently improves top-k precision in production RAG systems,
  which is why re-ranking is a standard second stage after hybrid
  retrieval, not a replacement for it. RRF gives us a strong, fast
  candidate set; the cross-encoder is a slower, more accurate filter
  applied only to that small candidate set (cheap because top_k is small).

Industry-grade properties:
  - Model loaded once, cached as a module-level singleton (loading is slow).
  - Defensive against empty input (empty candidate list -> empty result,
    no crash on the next pipeline stage).
  - Logs before/after ranking so a reviewer can see re-ranking's actual
    effect (e.g., did rank 5 jump to rank 1?) for debugging or demos.
"""
from sentence_transformers import CrossEncoder
from loguru import logger

from config import settings
from retrieval.hybrid_retrieval import RetrievedDoc

_reranker: CrossEncoder | None = None
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        logger.info(f"Loading cross-encoder reranker: {RERANKER_MODEL_NAME}")
        _reranker = CrossEncoder(RERANKER_MODEL_NAME)
    return _reranker


def rerank(query: str, docs: list[RetrievedDoc], top_k: int | None = None) -> list[RetrievedDoc]:
    """
    Re-scores `docs` against `query` using a cross-encoder and returns the
    top_k re-ranked results. Mutates each doc's `rerank_score` field in
    place (matching the pattern used for dense_rank/sparse_rank/rrf_score
    in hybrid_retrieval.py) so downstream callers can inspect the full
    scoring trail for any chunk.
    """
    top_k = top_k or settings.rerank_top_k

    if not docs:
        logger.warning("rerank() called with empty doc list — returning empty result.")
        return []

    model = get_reranker()
    pairs = [(query, d.text) for d in docs]
    scores = model.predict(pairs)

    for doc, score in zip(docs, scores):
        doc.rerank_score = float(score)

    ranked = sorted(docs, key=lambda d: d.rerank_score, reverse=True)
    result = ranked[:top_k]

    # Log the before/after ordering shift — useful for demos and debugging.
    before_order = [f"{d.source} p.{d.page}" for d in docs[:top_k]]
    after_order = [f"{d.source} p.{d.page}" for d in result]
    if before_order != after_order:
        logger.debug(
            f"Re-ranking changed top-{top_k} order.\n"
            f"  Before (RRF order):     {before_order}\n"
            f"  After (cross-encoder):  {after_order}"
        )
    else:
        logger.debug(f"Re-ranking confirmed existing top-{top_k} order (no change).")

    return result