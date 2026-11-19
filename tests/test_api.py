import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from app import encoder, settings  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(settings, "WARMUP", False)
    monkeypatch.setattr(encoder, "is_ready", lambda: True)
    monkeypatch.setattr(encoder, "encode", lambda texts: [[0.1, 0.2] for _ in texts])
    return TestClient(app)


def test_health_is_up_even_while_loading(monkeypatch):
    monkeypatch.setattr(encoder, "is_ready", lambda: False)
    c = TestClient(app)
    assert c.get("/health").status_code == 200
    assert c.get("/health/ready").status_code == 503


def test_embed_returns_dense_and_sparse_per_text(client):
    body = client.post("/embed", json={"texts": ["mức lương", "thuế"], "kind": "document"}).json()
    assert len(body["dense"]) == 2 and len(body["sparse"]) == 2
    assert body["sparse"][0]["indices"] and len(body["sparse"][0]["indices"]) == len(body["sparse"][0]["values"])


def test_query_sparse_weights_are_binary(client):
    body = client.post("/embed", json={"texts": ["mức lương"], "kind": "query"}).json()
    assert set(body["sparse"][0]["values"]) == {1.0}


def test_embed_503_while_model_loads(monkeypatch):
    monkeypatch.setattr(encoder, "is_ready", lambda: False)
    r = TestClient(app).post("/embed", json={"texts": ["x"]})
    assert r.status_code == 503 and r.headers["retry-after"] == "5"


def test_api_key_is_enforced_when_configured(client, monkeypatch):
    monkeypatch.setattr(settings, "API_KEY", "s3cret")
    assert client.post("/embed", json={"texts": ["x"]}).status_code == 401
    assert client.post("/embed", json={"texts": ["x"]}, headers={"X-API-Key": "s3cret"}).status_code == 200


def test_validation(client):
    assert client.post("/embed", json={"texts": []}).status_code == 422
    assert client.post("/embed", json={"texts": ["x"], "kind": "other"}).status_code == 422
