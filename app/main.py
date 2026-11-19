"""Embedding service. Run: uvicorn app.main:app --port 8000   (docs at /docs)

POST /embed  {"texts": [...], "kind": "query" | "document"}
  -> dense vectors (bge-m3, normalised) + sparse BM25-style vectors, so no other service needs the model
     or the tokenizer: the query API and the ingestion job both call this one.
"""
import asyncio
import secrets
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse

from app import encoder, settings
from app.schemas import EmbedRequest, EmbedResponse, SparseVec
from app.sparse import doc_vector, query_vector

_gate = asyncio.Semaphore(settings.MAX_CONCURRENCY)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.WARMUP:
        # Background load: the port opens at once (cloud health probes pass), /health/ready flips later.
        app.state.load_task = asyncio.create_task(asyncio.to_thread(encoder.load))
    yield


app = FastAPI(title="TrustRAG embedding service", version="1.0.0", lifespan=lifespan)


def require_key(x_api_key: str = Header(default="")) -> None:
    if settings.API_KEY and not secrets.compare_digest(x_api_key.encode(), settings.API_KEY.encode()):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


@app.get("/health")
async def health():
    """Liveness: the process is up (the model may still be loading)."""
    return {"status": "ok"}


@app.get("/health/ready")
async def ready():
    if encoder.is_ready():
        return {"status": "ready", "model": settings.MODEL, "dim": settings.DIM}
    return JSONResponse(status_code=503, content={"status": "loading", "error": encoder.error})


@app.post("/embed", response_model=EmbedResponse, dependencies=[Depends(require_key)])
async def embed(req: EmbedRequest):
    if not encoder.is_ready():
        detail = encoder.error or "Model is still loading, retry in a few seconds"
        raise HTTPException(status_code=503, detail=detail, headers={"Retry-After": "5"})
    texts = [t[: settings.MAX_CHARS] for t in req.texts]
    async with _gate:  # one encode at a time: parallel CPU encodes only slow each other down
        dense = await asyncio.to_thread(encoder.encode, texts)
    to_sparse = query_vector if req.kind == "query" else doc_vector
    sparse = []
    for t in texts:
        indices, values = to_sparse(t)
        sparse.append(SparseVec(indices=indices, values=values))
    return EmbedResponse(model=settings.MODEL, dim=settings.DIM, dense=dense, sparse=sparse)
