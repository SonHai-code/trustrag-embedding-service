"""Owns the dense model. Heavy imports stay inside load() so tests and the health endpoint need no torch."""
import logging

from app import settings

log = logging.getLogger("uvicorn.error")
_model = None
error: str | None = None


def load() -> None:
    global _model, error
    try:
        from sentence_transformers import SentenceTransformer
        log.info("Loading %s ...", settings.MODEL)
        _model = SentenceTransformer(settings.MODEL)
        log.info("Model ready")
    except Exception as exc:  # keep the server alive: /health/ready reports the failure
        error = f"{type(exc).__name__}: {exc}"
        log.exception("Model failed to load")


def is_ready() -> bool:
    return _model is not None


def encode(texts: list[str]) -> list[list[float]]:
    """Blocking and CPU-bound: call it from a thread."""
    vectors = _model.encode(texts, batch_size=settings.ENCODE_BATCH, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()
