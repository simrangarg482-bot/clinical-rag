from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from loguru import logger
from rag_pipeline.graph import ask

app = FastAPI(title="Clinical RAG API")
STATIC_DIR = Path(__file__).parent / "static"

class AskRequest(BaseModel):
    query: str
    web_search_enabled: bool = False

class SourceOut(BaseModel):
    source: str
    page: int
    rerank_score: float
    text: str

class WebSourceOut(BaseModel):
    title: str
    url: str
    content: str

class AskResponse(BaseModel):
    answer: str | None = None
    blocked_reason: str | None = None
    retries: int = 0
    support: dict | None = None
    sources: list[SourceOut] = []
    web_sources: list[WebSourceOut] = []
    source_type: str = "corpus"

@app.post("/api/ask", response_model=AskResponse)
def api_ask(req: AskRequest) -> AskResponse:
    logger.info(f"API request: '{req.query}' web_search={req.web_search_enabled}")
    state = ask(req.query, web_search_enabled=req.web_search_enabled)

    if state.get("blocked_reason"):
        return AskResponse(blocked_reason=state["blocked_reason"], retries=state["retries"])

    result = state.get("self_rag_result") or {}

    sources = [
        SourceOut(source=doc.source, page=doc.page, rerank_score=doc.rerank_score or 0.0, text=doc.text)
        for doc in result.get("relevant_docs", [])
    ]

    web_sources = [
        WebSourceOut(title=s.get("title",""), url=s.get("url",""), content=s.get("content",""))
        for s in result.get("web_sources", [])
    ]

    return AskResponse(
        answer=state.get("final_answer"),
        retries=state["retries"],
        support=result.get("support"),
        sources=sources,
        web_sources=web_sources,
        source_type=state.get("source_type", "corpus"),
    )

@app.get("/")
def serve_index():
    return FileResponse(STATIC_DIR / "index.html")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")