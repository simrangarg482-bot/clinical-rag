"""
Hybrid retrieval: dense (HyDE + Qdrant) + sparse (BM25) fused via
Reciprocal Rank Fusion (RRF).

Design decision — in-memory BM25:
  BM25 runs over the FULL corpus, independently of dense search, then the
  two ranked lists are fused. This is the textbook hybrid search pattern
  (dense catches semantic matches, sparse catches exact keyword/drug-name/
  dosage matches that embeddings can blur).

  The BM25 index is built once at process startup by pulling all points
  out of Qdrant and tokenizing them in memory. This is simple and fast up
  to roughly 50k chunks; beyond that, swap to Qdrant's native sparse
  vector support (FastEmbed/SPLADE) so BM25-equivalent scoring lives in
  the same DB instead of a separate in-memory structure. Documented here
  so this tradeoff is explicit, not accidental.

Industry-grade properties:
  - BM25 index built lazily once, cached as a module-level singleton.
  - Qdrant search retried on transient failure.
  - RRF fusion uses the standard k=60 smoothing constant.
  - Every retrieved doc carries its dense_rank, sparse_rank, and rrf_score
    in the payload — essential for debugging "why was this chunk ranked here".
"""
from dataclasses import dataclass, field

from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger
from config import get_qdrant_client

from config import settings
from retrieval.hyde import hyde_embed

RRF_K = 60  # standard smoothing constant from the RRF paper
MAX_CORPUS_FOR_INMEMORY_BM25 = 50_000  # documented scale ceiling, see module docstring


@dataclass
class RetrievedDoc:
    text: str
    source: str
    page: int
    chunk_index: int
    dense_score: float = 0.0
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rrf_score: float = 0.0
    relevance: str | None = None  # filled in later by Self-RAG
    rerank_score: float | None = None  # filled in later by reranker


class _BM25Index:
    """Lazily-built, process-wide BM25 index over the full Qdrant corpus."""

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._docs: list[RetrievedDoc] = []

    def _build(self) -> None:
        client = get_qdrant_client(timeout=60)
        count = client.count(settings.qdrant_collection).count
        if count == 0:
            raise RuntimeError(
                f"Collection '{settings.qdrant_collection}' is empty. "
                f"Run ingestion first: python -m data.ingest"
            )
        if count > MAX_CORPUS_FOR_INMEMORY_BM25:
            logger.warning(
                f"Corpus has {count} chunks, above the {MAX_CORPUS_FOR_INMEMORY_BM25} "
                f"in-memory BM25 ceiling. Consider switching to Qdrant sparse vectors."
            )

        logger.info(f"Building in-memory BM25 index over {count} chunks...")
        points, _ = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=count,
            with_payload=True,
            with_vectors=False,
        )
        self._docs = [
            RetrievedDoc(
                text=p.payload["text"],
                source=p.payload["source"],
                page=p.payload.get("page", -1),
                chunk_index=p.payload.get("chunk_index", -1),
            )
            for p in points
        ]
        tokenized = [d.text.lower().split() for d in self._docs]
        self._bm25 = BM25Okapi(tokenized)
        logger.success(f"BM25 index ready ({len(self._docs)} docs).")

    def search(self, query: str, top_k: int) -> list[tuple[RetrievedDoc, int]]:
        """Returns [(doc, rank)] sorted by BM25 score, rank starting at 1."""
        if self._bm25 is None:
            self._build()
        scores = self._bm25.get_scores(query.lower().split())
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self._docs[i], rank + 1) for rank, i in enumerate(ranked_idx)]


_bm25_index = _BM25Index()  # module-level singleton, built on first use


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=6))
def _dense_search(query_vector: list[float], top_k: int):
    client = get_qdrant_client(timeout=60)
    return client.search(
        collection_name=settings.qdrant_collection,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,
    )


def hybrid_retrieve(query: str, top_k: int | None = None) -> list[RetrievedDoc]:
    """
    Runs dense (HyDE-expanded) and sparse (BM25) search over the full
    corpus independently, fuses with RRF, and returns a deduped, ranked list.
    """
    top_k = top_k or settings.retrieval_top_k
    logger.debug(f"Hybrid retrieve for query: '{query}' (top_k={top_k})")

    # --- Dense leg ---
    hyde_vector, hypo_doc = hyde_embed(query)
    dense_hits = _dense_search(hyde_vector, top_k)
    dense_docs: dict[str, RetrievedDoc] = {}
    for rank, hit in enumerate(dense_hits, start=1):
        key = f"{hit.payload['source']}|{hit.payload.get('chunk_index', -1)}"
        dense_docs[key] = RetrievedDoc(
            text=hit.payload["text"],
            source=hit.payload["source"],
            page=hit.payload.get("page", -1),
            chunk_index=hit.payload.get("chunk_index", -1),
            dense_score=hit.score,
            dense_rank=rank,
        )

    # --- Sparse leg ---
    sparse_hits = _bm25_index.search(query, top_k)
    sparse_ranks: dict[str, int] = {}
    sparse_docs: dict[str, RetrievedDoc] = {}
    for doc, rank in sparse_hits:
        key = f"{doc.source}|{doc.chunk_index}"
        sparse_ranks[key] = rank
        sparse_docs[key] = doc

    # --- Reciprocal Rank Fusion over the union of both result sets ---
    all_keys = set(dense_docs) | set(sparse_docs)
    fused: list[RetrievedDoc] = []
    for key in all_keys:
        doc = dense_docs.get(key) or sparse_docs[key]
        dense_rank = dense_docs[key].dense_rank if key in dense_docs else None
        sparse_rank = sparse_ranks.get(key)

        score = 0.0
        if dense_rank is not None:
            score += 1.0 / (RRF_K + dense_rank)
        if sparse_rank is not None:
            score += 1.0 / (RRF_K + sparse_rank)

        doc.dense_rank = dense_rank
        doc.sparse_rank = sparse_rank
        doc.rrf_score = score
        fused.append(doc)

    fused.sort(key=lambda d: d.rrf_score, reverse=True)
    result = fused[:top_k]

    logger.debug(
        f"Hybrid retrieve returned {len(result)} docs "
        f"(dense_hits={len(dense_docs)}, sparse_hits={len(sparse_docs)}, "
        f"union={len(all_keys)}). HyDE passage used: '{hypo_doc[:80]}...'"
    )
    return result