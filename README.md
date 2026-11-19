# trustrag-embedding-service

One job: turn text into vectors. Used by the query API (question -> vector) and by the ingestion job (chunks -> vectors),
so the 2 GB model lives in exactly one place and can be scaled independently.

| Endpoint | Purpose |
|---|---|
| `POST /embed` | `{"texts": [...], "kind": "query"\|"document"}` -> `dense` (bge-m3, 1024-d, normalised) + `sparse` (BM25-style) per text |
| `GET /health` | liveness: process is up |
| `GET /health/ready` | 200 only after the model is loaded (503 while loading) |

The model loads in the background so the port opens immediately (cloud health probes pass); `/embed` answers 503 +
`Retry-After` until ready and callers retry.

```bash
pip install -r requirements.txt && uvicorn app.main:app --port 8001      # local
pip install -r requirements-dev.txt && pytest -q                         # tests (no model needed)
docker build -t trustrag-embedding-service .                             # image with the model baked in
```

Settings: see `.env.example`. `SERVICE_API_KEY` (optional) requires an `X-API-Key` header.
Resources: 2 vCPU / 4 GiB. Scale out for ingestion bursts; keep >= 1 replica for query latency.
