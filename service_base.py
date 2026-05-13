import logging

import torch
import torch.nn.functional as F
from fastapi import FastAPI
from transformers import AutoModel, AutoTokenizer

from src.config import MODEL_NAME
from src.schemas import EmbedRequest, EmbedResponse

logger = logging.getLogger(__name__)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

app = FastAPI(title='Part 1 — Base Inference')


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/embed', response_model=EmbedResponse)
def embed(req: EmbedRequest):
    encoded = tokenizer(  # ty:ignore[call-non-callable]
        req.texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt',
    )
    with torch.no_grad():
        output = model(**encoded)
    embs = output.last_hidden_state[:, 0]  # CLS token
    embs = F.normalize(embs, p=2, dim=1)
    return EmbedResponse(embeddings=embs.tolist())
