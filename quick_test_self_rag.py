from retrieval.hybrid_retrieval import hybrid_retrieve
from retrieval.reranker import rerank
from rag_pipeline.self_rag import self_rag_pipeline

query = "what is the recommended first-line treatment for hypertension?"

docs = hybrid_retrieve(query)
top_docs = rerank(query, docs)

result = self_rag_pipeline(query, top_docs)

print(f"\nStatus: {result['status']}")
if result['status'] == 'success':
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSupport verdict: {result['support']}")
    print(f"\nUsed {len(result['relevant_docs'])} relevant docs out of {len(top_docs)} candidates")
else:
    print(f"Only {len(result['relevant_docs'])} relevant docs found — would trigger Corrective RAG")