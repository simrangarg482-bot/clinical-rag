"""
Central config for the Clinical RAG project .

Why pydantic-settings instead of plain os.getenv():
  - Fails FAST and LOUD at startup if config is missing/wrong-typed,
    instead of failing silently 10 steps later with a cryptic API error.
  - Self-documenting: every config value has a type and a description.
  - This is the pattern used in production FastAPI/backend services.
"""
import sys
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from loguru import logger
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from qdrant_client import QdrantClient


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- OpenRouter ---
    openrouter_api_key: str = Field(..., description="OpenRouter API key, required")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1")
    llm_model: str = Field(default="openai/gpt-4o")
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=1024, gt=0)

    # --- Qdrant ---
    qdrant_host: str = Field(default="localhost")
    qdrant_port: int = Field(default=6333)
    qdrant_collection: str = Field(default="clinical_docs")

    # --- Embeddings ---
    embedding_model_name: str = Field(default="all-MiniLM-L6-v2")
    embedding_dim: int = Field(default=384)

    # --- Retrieval tuning ---
    retrieval_top_k: int = Field(default=18, gt=0)
    rerank_top_k: int = Field(default=3, gt=0)
    max_self_rag_retries: int = Field(default=2, ge=0)

    # --- Environment ---
    environment: str = Field(default="development")
    log_level: str = Field(default="INFO")

    # --- Web search fallback ---
    tavily_api_key: str = Field(default="", description="Tavily API key for web search fallback")
    web_search_enabled: bool = Field(default=False)

    qdrant_api_key: str = Field(default="", description="Qdrant Cloud API key, empty for local")
    qdrant_use_https: bool = Field(default=False)

    @field_validator("openrouter_api_key")
    @classmethod
    def key_not_placeholder(cls, v: str) -> str:
        if not v or "your_openrouter_key" in v:
            raise ValueError(
                "OPENROUTER_API_KEY is missing. "
            )
        return v


@lru_cache  # singleton — config is parsed once per process, not on every import
def get_settings() -> Settings:
    try:
        return Settings()
    except Exception as e:
        # Fail fast and loud — this is what you want at startup, not 10 calls deep
        logger.critical(f"Configuration error: {e}")
        sys.exit(1)


settings = get_settings()

# --- Structured logging setup ---
logger.remove()
logger.add(
    sys.stderr,
    level=settings.log_level,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
           "<cyan>{module}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
)
logger.add(
    "logs/clinical_rag_{time:YYYY-MM-DD}.log",
    rotation="00:00",
    retention="14 days",
    level="DEBUG",
    backtrace=True,
)


def get_qdrant_client(timeout: int = 30) -> QdrantClient:
    print(f"DEBUG qdrant_host: '{settings.qdrant_host}'")
    print(f"DEBUG qdrant_api_key set: {bool(settings.qdrant_api_key)}")
    print(f"DEBUG full url: 'https://{settings.qdrant_host}'")

    if settings.qdrant_api_key:
        return QdrantClient(
        url=f"https://{settings.qdrant_host.removeprefix('https://').removeprefix('http://')}",
        api_key=settings.qdrant_api_key,
        timeout=timeout,
    )

    return QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        timeout=timeout,
    )


def get_llm(temperature: float | None = None) -> ChatOpenAI:
    """Returns a LangChain chat model pointed at OpenRouter."""
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=temperature if temperature is not None else settings.llm_temperature,
        openai_api_key=settings.openrouter_api_key,
        openai_api_base=settings.openrouter_base_url,
        max_tokens=settings.llm_max_tokens,
        default_headers={
            "HTTP-Referer": "http://localhost",
            "X-Title": "Clinical RAG Project",
        },
        max_retries=0,  # we handle retries ourselves below, with logging
        request_timeout=30,
    )


# A single shared LLM instance reused across modules (avoids re-instantiating per call)
_llm = get_llm() #Underscore means: This is internal, not meant to be imported everywhere


def call_llm(prompt: str, temperature: float | None = None, system_prompt: str | None = None) -> str:
    """
    The ONE function every module should call to hit the LLM.
    Wraps every LLM call in exponential backoff retry — production systems
    never call an external API directly without this, since transient
    network errors / 429s / 5xxs are routine, not exceptional.

    Free-tier OpenRouter models occasionally return 429 with a
    retry_after_seconds hint around 15-30s when the shared upstream pool is
    saturated — 5 attempts with a 30s ceiling gives those a real chance to
    clear, instead of giving up after a couple of quick retries.

    Also guards against response.content being None or empty — some
    free-tier models occasionally return a malformed/empty completion
    instead of raising an error. We treat that as a retryable failure
    (via the inner _call) rather than letting every caller's .strip()
    crash with AttributeError.

    system_prompt: optional framing message. Clinical-content prompts
    (grading relevance of a medical passage, checking groundedness of a
    clinical answer) occasionally trip a model's built-in safety filter
    for "unauthorized medical advice" even though the task is evaluation
    over reference text, not advice to a patient. Pass CLINICAL_SYSTEM_PROMPT
    (defined below) for any call operating on clinical guideline content
    to reduce false-positive refusals.
    """

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=4, max=30),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: logger.warning(
            f"LLM call failed (attempt {retry_state.attempt_number}/5): "
            f"{type(retry_state.outcome.exception()).__name__}: "
            f"{retry_state.outcome.exception()} -- retrying in "
            f"{retry_state.next_action.sleep:.1f}s..."
        ),
    )
    def _call() -> str:
        llm = _llm if temperature is None else get_llm(temperature)
        if system_prompt:
            from langchain_core.messages import SystemMessage, HumanMessage
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]
            response = llm.invoke(messages)
        else:
            response = llm.invoke(prompt)
        content = response.content
        if not content or not isinstance(content, str) or not content.strip():
            logger.debug(f"Empty content. Full response object: {response!r}")
            raise ValueError(
                f"LLM returned empty/invalid content (got: {content!r}). "
                f"Treating as a transient failure to trigger retry."
            )
        return content

    return _call()


CLINICAL_SYSTEM_PROMPT = (
    "You are a clinical informatics evaluation assistant operating ONLY on "
    "published clinical guideline text that has already been retrieved from "
    "a vetted document corpus. Your task is structured evaluation and "
    "summarization of this reference material — grading relevance, "
    "checking whether a summary is grounded in its source text, or "
    "generating a citation-backed summary of the provided passages. "
    "You are NOT diagnosing, treating, or advising any individual patient; "
    "you are processing reference documents for a clinical decision support "
    "tool used by licensed clinicians. Follow the requested output format exactly."
)

# --- Local embedding model ---
_embedder: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        logger.info(f"Loading embedding model: {settings.embedding_model_name}")
        _embedder = SentenceTransformer(settings.embedding_model_name)
    return _embedder


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedder()
    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]