from pydantic import BaseModel, Field

from src.config import MAX_BATCH_SIZE


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
