from retrieval.hybrid_retrieval import hybrid_retrieve
from retrieval.reranker import rerank

query = "what is the recommended first-line treatment for hypertension?"

docs = hybrid_retrieve(query)
print(f"\n--- Top 5 after RRF fusion (before re-ranking) ---")
for d in docs[:5]:
    print(f"[rrf={d.rrf_score:.4f}] {d.source} p.{d.page}: {d.text[:80]}...")

reranked = rerank(query, docs)
print(f"\n--- Top {len(reranked)} after cross-encoder re-ranking ---")
for d in reranked:
    print(f"[rerank={d.rerank_score:.4f} rrf={d.rrf_score:.4f}] {d.source} p.{d.page}: {d.text[:80]}...")