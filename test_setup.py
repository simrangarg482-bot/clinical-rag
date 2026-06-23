"""
Sanity check: confirms OpenRouter LLM, local embedder, and Qdrant are all reachable.
Run once after setup, and again any time you change .env.
"""
from config import settings, call_llm, embed_query
from qdrant_client import QdrantClient
from loguru import logger
from config import get_qdrant_client

logger.info("Step 1/3 — testing OpenRouter LLM connection...")
result = call_llm("Reply with exactly one word: OK")
logger.success(f"LLM responded: {result.strip()}")

logger.info("Step 2/3 — testing local embedder...")
vec = embed_query("test sentence")
logger.success(f"Embedding generated, length = {len(vec)} (expected {settings.embedding_dim})")
assert len(vec) == settings.embedding_dim, "Embedding dim mismatch with config!"

logger.info("Step 3/3 — testing Qdrant connection...")
client = get_qdrant_client(timeout=60)
collections = client.get_collections()
logger.success(f"Qdrant connected. Existing collections: {collections.collections}")

logger.success("All systems go. Ready for Step 2 (data ingestion).")