"""
RAGAS evaluation harness.

Measures the four standard RAG quality metrics against a fixed benchmark
dataset, so retrieval/generation quality is tracked with numbers instead
of spot-checking individual queries by eye.

  - faithfulness:        is the generated answer grounded in the retrieved
                          context (no hallucinated claims)?
  - answer_relevancy:    does the answer actually address the question?
  - context_precision:   of the retrieved chunks, how many were relevant?
  - context_recall:      did retrieval find the chunks needed to answer
                          fully, against a ground-truth reference?

Industry-grade properties:
  - The benchmark dataset is VERSIONED (a JSON file checked into the repo,
    not hardcoded in this script) — so eval runs are reproducible and the
    dataset can grow over time without touching code.
  - Each run's results are saved with a timestamp to evaluation/results/,
    so quality can be tracked across changes to the pipeline (e.g. "did
    swapping the reranker model improve faithfulness?").
  - Uses the project's own OpenRouter-backed LLM as RAGAS's judge model,
    instead of requiring a separate OpenAI key, consistent with the rest
    of the project running on a single provider.
"""
import json
from datetime import datetime
from pathlib import Path

from datasets import Dataset
from loguru import logger
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

from config import settings
from rag_pipeline.graph import ask

BENCHMARK_PATH = Path(__file__).parent / "benchmark_dataset.json"
RESULTS_DIR = Path(__file__).parent / "results"


def _build_ragas_llm_wrapper():
    """
    RAGAS's metrics need an LLM to act as the judge (e.g. to check if an
    answer's claims are entailed by the context). By default RAGAS reaches
    for OPENAI_API_KEY directly — this wraps our existing OpenRouter-backed
    ChatOpenAI client instead, so the whole project stays on one provider
    and one set of credentials.
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from config import get_llm

    ragas_llm = LangchainLLMWrapper(get_llm(temperature=0.0))
    ragas_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=settings.embedding_model_name)
    )
    return ragas_llm, ragas_embeddings


def load_benchmark() -> list[dict]:
    """
    Loads the versioned benchmark dataset.
    Each entry: {"question": str, "ground_truth": str}
    ground_truth is a hand-written reference answer used for context_recall.
    """
    if not BENCHMARK_PATH.exists():
        raise FileNotFoundError(
            f"Benchmark dataset not found at {BENCHMARK_PATH}. "
            f"Create it first (see evaluation/benchmark_dataset.json)."
        )
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def run_pipeline_on_benchmark(benchmark: list[dict]) -> list[dict]:
    """
    Runs each benchmark question through the actual LangGraph pipeline,
    collecting question/answer/contexts/ground_truth in RAGAS's expected
    shape. This is what makes the eval honest — it exercises the REAL
    system end-to-end, not a mocked-out shortcut.
    """
    rows = []
    for i, item in enumerate(benchmark, start=1):
        question = item["question"]
        logger.info(f"[{i}/{len(benchmark)}] Running pipeline for: '{question}'")

        state = ask(question)

        if state.get("blocked_reason"):
            logger.warning(f"Query blocked, skipping from eval: {state['blocked_reason']}")
            continue

        result = state.get("self_rag_result") or {}
        contexts = [doc.text for doc in result.get("relevant_docs", [])]
        answer = state.get("final_answer") or ""

        if not contexts:
            logger.warning(f"No contexts retrieved for '{question}' — RAGAS scores will be degenerate.")

        rows.append({
            "question": question,
            "answer": answer,
            "contexts": contexts if contexts else [""],  # RAGAS requires non-empty contexts list
            "ground_truth": item["ground_truth"],
        })
    return rows


def run_ragas_eval(rows: list[dict]) -> dict:
    """Runs RAGAS's four core metrics over the collected pipeline outputs."""
    ragas_llm, ragas_embeddings = _build_ragas_llm_wrapper()

    for metric in (faithfulness, answer_relevancy, context_precision, context_recall):
        metric.llm = ragas_llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = ragas_embeddings

    dataset = Dataset.from_list(rows)
    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    return results


def save_results(results: dict, rows: list[dict]) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"eval_{timestamp}.json"

    summary = {
        "timestamp": timestamp,
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model_name,
        "n_questions": len(rows),
        "scores": {k: float(v) for k, v in results.items()},
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.success(f"Results saved to {out_path}")
    return out_path


def main():
    benchmark = load_benchmark()
    logger.info(f"Loaded {len(benchmark)} benchmark questions.")

    rows = run_pipeline_on_benchmark(benchmark)
    if not rows:
        logger.error("No usable rows produced — every query was blocked. Aborting eval.")
        return

    logger.info(f"Running RAGAS evaluation over {len(rows)} pipeline outputs...")
    results = run_ragas_eval(rows)

    print("\n===== RAGAS Evaluation Results =====")
    for metric, score in results.items():
        print(f"{metric:20s}: {score:.3f}")

    save_results(results, rows)


if __name__ == "__main__":
    main()