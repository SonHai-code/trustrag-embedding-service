"""Settings, read from environment variables (see .env.example)."""
import os

MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
MAX_BATCH = int(os.getenv("MAX_BATCH", "64"))          # texts per request
MAX_CHARS = int(os.getenv("MAX_CHARS", "8000"))        # longer texts are truncated
ENCODE_BATCH = int(os.getenv("ENCODE_BATCH", "16"))    # model batch size (memory vs speed)
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "1"))  # encodes running at once; CPU-bound, so keep it 1
API_KEY = os.getenv("SERVICE_API_KEY", "")             # optional shared secret, sent as X-API-Key
WARMUP = os.getenv("WARMUP", "true").lower() == "true"  # load the model in the background at startup
