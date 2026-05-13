import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from transformers import AutoTokenizer

from src.batch_queue import BatchQueue
from src.config import MODEL_NAME, ONNX_PATH
from src.embeddings import create_onnx_session, onnx_embed
from src.schemas import EmbedRequest, EmbedResponse

logger = logging.getLogger(__name__)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
session = create_onnx_session(ONNX_PATH)

batch_queue = BatchQueue(infer_fn=lambda texts: onnx_embed(session, tokenizer, texts))


@asynccontextmanager
async def lifespan(app: FastAPI):
    await batch_queue.start()
    yield
    await batch_queue.stop()


app = FastAPI(title='Part 3 — ONNX + Dynamic Batching', lifespan=lifespan)


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/embed', response_model=EmbedResponse)
async def embed(req: EmbedRequest):
    embeddings = await batch_queue.submit(req.texts)
    return EmbedResponse(embeddings=embeddings)
