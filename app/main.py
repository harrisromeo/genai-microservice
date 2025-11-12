from __future__ import annotations

import time
from typing import Any, Final

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import PlainTextResponse, JSONResponse

from app.config import settings
from app.schemas import ChatRequest, ChatResponse, EmbedRequest, EmbedResponse

# ---- Metrics ---------------------------------------------------------------
REQUESTS_TOTAL: Final = Counter(
    "requests_total", "Total HTTP requests", ["path", "method", "code"]
)
LATENCY_MS: Final = Histogram(
    "request_latency_ms", "Request latency in ms", ["path", "method"], buckets=(5, 10, 25, 50, 100, 250, 500, 1000)
)

# ---- App -------------------------------------------------------------------
app = FastAPI(title="GenAI Microservice", version="1.0.0")


def _validate_chat(req: ChatRequest) -> None:
    if len(req.prompt) > settings.prompt_max_chars:
        raise HTTPException(status_code=400, detail="prompt too long")
    if req.max_tokens > settings.max_tokens_limit:
        raise HTTPException(status_code=400, detail="max_tokens exceeds limit")
    if req.max_tokens <= 0:
        raise HTTPException(status_code=400, detail="max_tokens must be positive")


async def _dummy_generate_chat(prompt: str, max_tokens: int) -> str:
    # Simple echo for now; keeps container booting without external keys.
    return prompt[: max_tokens] if max_tokens < len(prompt) else prompt


def _dummy_embed(texts: list[str]) -> list[list[float]]:
    # 3-D toy vectors; deterministic mapping by length
    out: list[list[float]] = []
    for t in texts:
        n = float(len(t))
        out.append([n % 7.0, n % 3.0, 1.0])
    return out


# ---- Routes ----------------------------------------------------------------
@app.get("/")
async def root() -> dict[str, str]:
    return {"status": "ok", "message": "GenAI microservice running. See /docs."}


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "healthy"}


@app.head("/healthz")
async def healthz_head() -> Response:
    return Response(status_code=200)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    t0 = time.perf_counter()
    code = "200"
    try:
        _validate_chat(req)
        text = await _dummy_generate_chat(req.prompt, req.max_tokens)
        return ChatResponse(
            text=text,
            input_tokens=len(req.prompt.split()),
            output_tokens=req.max_tokens,
            model=settings.model_name,
        )
    except HTTPException as e:
        code = str(e.status_code)
        raise
    finally:
        LATENCY_MS.labels("/chat", "POST").observe((time.perf_counter() - t0) * 1000.0)
        REQUESTS_TOTAL.labels("/chat", "POST", code).inc()


@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest) -> EmbedResponse:
    t0 = time.perf_counter()
    code = "200"
    try:
        if not req.texts:
            raise HTTPException(status_code=400, detail="texts must not be empty")
        vectors = _dummy_embed(req.texts)
        return EmbedResponse(vectors=vectors, dim=3, model="dummy-embedder")
    except HTTPException as e:
        code = str(e.status_code)
        raise
    finally:
        LATENCY_MS.labels("/embed", "POST").observe((time.perf_counter() - t0) * 1000.0)
        REQUESTS_TOTAL.labels("/embed", "POST", code).inc()


@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    data = generate_latest()  # type: ignore[arg-type]
    return PlainTextResponse(data.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)

