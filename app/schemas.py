from typing import Literal

from pydantic import BaseModel, Field

from app import settings


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=settings.MAX_BATCH)
    kind: Literal["query", "document"] = "document"


class SparseVec(BaseModel):
    indices: list[int]
    values: list[float]


class EmbedResponse(BaseModel):
    model: str
    dim: int
    dense: list[list[float]]
    sparse: list[SparseVec]
