import numpy as np
import onnxruntime as ort


def create_onnx_session(onnx_path: str) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.enable_mem_pattern = False
    return ort.InferenceSession(onnx_path, sess_options=opts, providers=['CPUExecutionProvider'])


def cls_pooling(last_hidden_state: np.ndarray) -> np.ndarray:
    return last_hidden_state[:, 0]


def normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-9, None)


def onnx_embed(session, tokenizer, texts: list[str]) -> np.ndarray:
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='np')
    outputs = session.run(
        None,
        {'input_ids': encoded['input_ids'], 'attention_mask': encoded['attention_mask']},
    )
    embs = cls_pooling(outputs[0])
    return normalize(embs)
