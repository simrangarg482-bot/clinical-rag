# Clinical Decision Support RAG System

A production-grade Retrieval-Augmented Generation pipeline for clinical guideline Q&A, built as a portfolio project demonstrating advanced RAG techniques.

---

## Architecture

```
Query
  │
  ▼
Input Guardrail (PII check · scope check)
  │
  ▼
HyDE Query Expansion (hypothetical document embedding)
  │
  ▼
Hybrid Retrieval (dense cosine + BM25 sparse → RRF fusion)
  │
  ▼
Cross-Encoder Re-ranking
  │
  ▼
Self-RAG Relevance Filtering (parallel LLM grading per chunk)
  │
  ├── insufficient context → Corrective RAG (query rewrite + retry)
  │
  └── sufficient context →
        LLM Generation (grounded, cited answer)
          │
          ▼
        Support Check (hallucination detection)
          │
          ▼
        Output Guardrail (toxicity · hedging language)
          │
          ▼
        Final Answer + Source Citations
```

---

## Techniques Demonstrated

| Technique | Implementation |
|---|---|
| HyDE | `retrieval/hyde.py` — generates a hypothetical clinical passage, embeds it for better recall on vague queries |
| Hybrid Retrieval | `retrieval/hybrid_retrieval.py` — dense (all-MiniLM-L6-v2) + BM25 fused via Reciprocal Rank Fusion (k=60) |
| Cross-Encoder Re-ranking | `retrieval/reranker.py` — `ms-marco-MiniLM-L-6-v2` re-scores query-passage pairs jointly |
| Self-RAG | `rag_pipeline/self_rag.py` — LLM grades each retrieved chunk for relevance, then checks groundedness of the generated answer |
| Corrective RAG | `rag_pipeline/self_rag.py` — rewrites the query if fewer than 2 relevant chunks are found, retried up to `max_self_rag_retries` times |
| Agentic RAG (LangGraph) | `rag_pipeline/graph.py` — stateful graph with conditional routing (retry loop, guardrail short-circuits) |
| Guardrails | `rag_guardrails/` — `guardrails-ai` for PII detection (input) and toxicity checking (output) |
| RAGAS Evaluation | `evaluation/ragas_eval.py` — faithfulness, answer relevancy, context precision, context recall |

---

## Corpus

Currently indexes **4 WHO/CDC clinical guideline PDFs**:

- Hypertension pharmacological treatment guidelines
- Tuberculosis (TB) consolidated guidelines
- TB & diabetes comorbidity management
- HIV/comorbidity guidelines

**Corpus-scoped by design.** The system only answers from its ingested corpus and returns an explicit "insufficient context" message for out-of-scope queries — this is a deliberate trustworthiness decision, not a limitation. In a clinical context, confidently answering something unsupported is worse than saying "I don't know."

### Expanding the corpus

Drop additional PDF guidelines into `data/pdfs/` and re-run ingestion:

```bash
python -m data.ingest
```

The ingestion pipeline is **idempotent** — re-running on the same PDFs won't create duplicate vectors (deterministic SHA-256 chunk IDs). New PDFs are automatically picked up and indexed.

---

## Tech Stack

| Layer | Tool |
|---|---|
| LLM | OpenRouter (`gpt-4o-mini` recommended, any OpenAI-compatible model) |
| Orchestration | LangGraph |
| Vector DB | Qdrant (Docker) |
| Embeddings | `all-MiniLM-L6-v2` (local, CPU, no API key needed) |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local) |
| Guardrails | `guardrails-ai` with DetectPII + ToxicLanguage validators |
| Evaluation | RAGAS |
| API | FastAPI |
| Frontend | Vanilla HTML/CSS/JS (dark, Perplexity-inspired) |

---

## Setup

### Prerequisites
- Python 3.10+
- Docker Desktop

### 1. Start Qdrant

```bash
docker run -d -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  --restart unless-stopped \
  qdrant/qdrant
```

### 2. Install dependencies

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and add your OpenRouter API key
```

`.env` variables:

```env
OPENROUTER_API_KEY=your_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=openai/gpt-4o-mini
LLM_MAX_TOKENS=1024
QDRANT_HOST=127.0.0.1
QDRANT_PORT=6333
ENVIRONMENT=development
LOG_LEVEL=INFO
```

### 4. Install Guardrails Hub validators

```bash
guardrails configure   # paste your Hub token
guardrails hub install hub://guardrails/detect_pii
guardrails hub install hub://guardrails/toxic_language
```

### 5. Ingest documents

Drop clinical guideline PDFs into `data/pdfs/`, then:

```bash
python -m data.ingest
```

### 6. Run the app

```bash
uvicorn server:app --reload --port 8000
```

Open `http://127.0.0.1:8000`

---

## Evaluation

Run RAGAS evaluation against the benchmark dataset:

```bash
python -m evaluation.ragas_eval
```

Results are saved with timestamps to `evaluation/results/`.

---

## Project Structure

```
clinical-rag/
├── data/
│   ├── pdfs/               # Drop clinical guideline PDFs here
│   └── ingest.py           # Idempotent ingestion pipeline
├── retrieval/
│   ├── hyde.py             # HyDE query expansion
│   ├── hybrid_retrieval.py # Dense + BM25 + RRF fusion
│   └── reranker.py         # Cross-encoder re-ranking
├── rag_pipeline/
│   ├── self_rag.py         # Self-RAG + Corrective RAG
│   └── graph.py            # LangGraph orchestration
├── rag_guardrails/
│   ├── input_guard.py      # PII detection + scope check
│   └── output_guard.py     # Toxicity + hedging check
├── evaluation/
│   ├── ragas_eval.py       # RAGAS evaluation harness
│   └── benchmark_dataset.json
├── static/                 # Frontend (HTML/CSS/JS)
├── server.py               # FastAPI backend
├── config.py               # Pydantic settings + LLM client
└── requirements.txt
```

---

## Known Limitations & Future Work

- **Latency:** ~20-40s per query on free-tier models; ~5-10s with `gpt-4o-mini` on a paid key. The parallelized relevance checks help, but the sequential HyDE → generation → support-check stages are inherent to the Self-RAG architecture.
- **BM25 scale ceiling:** In-memory BM25 is practical up to ~50k chunks; beyond that, swap to Qdrant's native sparse vectors (documented in `hybrid_retrieval.py`).
- **Corpus scope:** Currently 4 guideline PDFs. Trivially expandable — drop new PDFs in `data/pdfs/` and re-run ingest.

---

## Resume Line

> **Clinical Decision Support RAG System** — Built an end-to-end agentic RAG pipeline over WHO/CDC clinical guidelines using Self-RAG critique tokens, Corrective RAG with automatic query rewriting, HyDE query expansion, hybrid retrieval (dense + BM25 + RRF), cross-encoder re-ranking, guardrails-ai for PII/toxicity filtering, and LangGraph for stateful agent orchestration. Parallelized relevance-grading stage for ~5x latency reduction. Evaluated with RAGAS (faithfulness, answer relevancy, context precision/recall). FastAPI backend + custom dark-mode frontend.
