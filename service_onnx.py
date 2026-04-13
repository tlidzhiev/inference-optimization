import logging

from fastapi import FastAPI
from transformers import AutoTokenizer

from src.config import MODEL_NAME, ONNX_PATH
from src.embeddings import create_onnx_session, onnx_embed
from src.schemas import EmbedRequest, EmbedResponse

logger = logging.getLogger(__name__)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
session = create_onnx_session(ONNX_PATH)

app = FastAPI(title='Part 2 — ONNX Inference')


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/embed', response_model=EmbedResponse)
def embed(req: EmbedRequest):
    embs = onnx_embed(session, tokenizer, req.texts)
    return EmbedResponse(embeddings=embs.tolist())
