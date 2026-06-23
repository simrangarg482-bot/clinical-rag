"""
End-to-end test of the full LangGraph pipeline.
Exercises: input guard -> retrieve -> rerank -> self-RAG -> (corrective RAG
if needed) -> output guard -> final answer.
"""
from rag_pipeline.graph import ask

TEST_QUERIES = [
    "What is the recommended first-line treatment for hypertension?",
    "What is the recommended treatment for diabetes and TB comorbidity?",
]

for query in TEST_QUERIES:
    print(f"\n{'=' * 70}")
    print(f"QUERY: {query}")
    print('=' * 70)

    state = ask(query)

    if state.get("blocked_reason"):
        print(f"\n[BLOCKED] {state['blocked_reason']}")
        continue

    print(f"\nFINAL ANSWER:\n{state['final_answer']}")

    result = state.get("self_rag_result")
    if result and result.get("status") == "success":
        print(f"\nSupport verdict: {result['support']}")
        print(f"Sources used: {len(result['relevant_docs'])}")
        for doc in result["relevant_docs"]:
            print(f"  - {doc.source} p.{doc.page} (rerank_score={doc.rerank_score:.2f})")

    print(f"\nCorrective RAG retries used: {state['retries']}")