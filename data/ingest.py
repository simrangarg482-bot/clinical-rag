"""
Data ingestion pipeline for the Clinical RAG project.

  - IDEMPOTENT: deterministic chunk IDs (hash of source + content) mean
    re-running ingestion on the same PDFs never creates duplicate vectors.
  - BATCHED: embeds and upserts in batches, so a 500-page PDF set doesn't
    spike memory or send one giant request.
  - RESILIENT: Qdrant upserts retry on transient failure.
  - TRACEABLE: every chunk's payload carries enough metadata (source file,
    page number, chunk index, embedding model version) to debug a bad
    retrieval result back to its exact origin later.
"""
import hashlib
from pathlib import Path

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger
from config import get_qdrant_client

from config import settings, embed_texts

BATCH_SIZE = 64  # embed + upsert in chunks of this size


def _deterministic_id(source: str, chunk_index: int, text: str) -> str:
    """
    Hash-based ID so re-ingesting the same PDF produces the SAME point IDs.
    Qdrant upsert is an UPSERT (insert-or-replace) keyed on ID — so this
    makes the whole pipeline idempotent: run it twice, get one copy of the data.
    """
    raw = f"{source}|{chunk_index}|{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _ensure_collection(client: QdrantClient) -> None:
    """Create the collection only if it doesn't already exist — safe to call every run."""
    existing = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection in existing:
        logger.info(f"Collection '{settings.qdrant_collection}' already exists, reusing it.")
        return
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=VectorParams(size=settings.embedding_dim, distance=Distance.COSINE),
    )
    logger.info(f"Created collection '{settings.qdrant_collection}' (dim={settings.embedding_dim}).")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
def _upsert_batch(client: QdrantClient, points: list[PointStruct]) -> None:
    client.upsert(collection_name=settings.qdrant_collection, points=points)


def ingest_documents(pdf_dir: str = "data/pdfs") -> dict:
    """
    Loads all PDFs in pdf_dir, chunks them, embeds, and upserts into Qdrant.
    Returns a summary dict — useful for logging/asserting in tests.
    """
    pdf_path = Path(pdf_dir)
    if not pdf_path.exists() or not any(pdf_path.glob("*.pdf")):
        raise FileNotFoundError(
            f"No PDFs found in '{pdf_dir}'. Drop clinical guideline PDFs there first."
        )

    logger.info(f"Loading PDFs from {pdf_dir}...")
    loader = PyPDFDirectoryLoader(pdf_dir)
    docs = loader.load()
    logger.info(f"Loaded {len(docs)} pages from {len(list(pdf_path.glob('*.pdf')))} PDF(s).")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(docs)
    logger.info(f"Split into {len(chunks)} chunks (size=512, overlap=64).")

    client = get_qdrant_client(timeout=60)
    _ensure_collection(client)

    total_upserted = 0
    # Track per-source chunk index for deterministic IDs and debuggability
    source_counters: dict[str, int] = {}

    for batch_start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[batch_start: batch_start + BATCH_SIZE]
        texts = [c.page_content for c in batch]

        vectors = embed_texts(texts)

        points = []
        for chunk, vector in zip(batch, vectors):
            source = chunk.metadata.get("source", "unknown")
            page = chunk.metadata.get("page", -1)
            idx = source_counters.get(source, 0)
            source_counters[source] = idx + 1

            point_id = _deterministic_id(source, idx, chunk.page_content)
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "text": chunk.page_content,
                        "source": Path(source).name,
                        "page": page,
                        "chunk_index": idx,
                        "embedding_model": settings.embedding_model_name,
                    },
                )
            )

        _upsert_batch(client, points)
        total_upserted += len(points)
        logger.info(
            f"Upserted batch {batch_start // BATCH_SIZE + 1} "
            f"({total_upserted}/{len(chunks)} chunks so far)..."
        )

    logger.success(
        f"Ingestion complete. {total_upserted} chunks indexed into "
        f"'{settings.qdrant_collection}' from {len(source_counters)} source file(s)."
    )
    return {
        "total_chunks": total_upserted,
        "sources": list(source_counters.keys()),
        "collection": settings.qdrant_collection,
    }


if __name__ == "__main__":
    summary = ingest_documents()
    logger.info(f"Summary: {summary}")