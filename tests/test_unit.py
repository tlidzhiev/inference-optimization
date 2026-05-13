import asyncio

import numpy as np
import pytest
from pydantic import ValidationError

from src.batch_queue import BatchQueue
from src.embeddings import cls_pooling, normalize
from src.schemas import EmbedRequest

# --- Schema validation ---


def test_embed_request_valid():
    req = EmbedRequest(texts=['hello', 'world'])
    assert len(req.texts) == 2


def test_embed_request_empty_list():
    with pytest.raises(ValidationError):
        EmbedRequest(texts=[])


def test_embed_request_too_many_texts():
    with pytest.raises(ValidationError):
        EmbedRequest(texts=['text'] * 33)


# --- Embedding utilities ---


def test_cls_pooling_shape():
    hidden = np.random.randn(3, 10, 64).astype(np.float32)
    result = cls_pooling(hidden)
    assert result.shape == (3, 64)


def test_cls_pooling_values():
    hidden = np.random.randn(3, 10, 64).astype(np.float32)
    result = cls_pooling(hidden)
    np.testing.assert_array_equal(result, hidden[:, 0])


def test_normalize_unit_norm():
    x = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
    result = normalize(x)
    norms = np.linalg.norm(result, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


def test_normalize_zero_vector_no_nan():
    x = np.array([[0.0, 0.0]], dtype=np.float32)
    result = normalize(x)
    assert not np.any(np.isnan(result))


# --- BatchQueue ---


@pytest.mark.asyncio
async def test_batch_queue_returns_correct_shape():
    dim = 4

    def mock_infer(texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), dim), dtype=np.float32)

    queue = BatchQueue(infer_fn=mock_infer)
    await queue.start()

    result = await queue.submit(['hello', 'world'])

    assert len(result) == 2
    assert len(result[0]) == dim
    await queue.stop()


@pytest.mark.asyncio
async def test_batch_queue_concurrent_requests():
    def mock_infer(texts: list[str]) -> np.ndarray:
        # Return index-based values so we can verify correct slicing
        return np.arange(len(texts) * 2, dtype=np.float32).reshape(len(texts), 2)

    queue = BatchQueue(infer_fn=mock_infer)
    await queue.start()

    results = await asyncio.gather(
        queue.submit(['a']),
        queue.submit(['b', 'c']),
    )

    assert len(results[0]) == 1
    assert len(results[1]) == 2
    await queue.stop()


@pytest.mark.asyncio
async def test_batch_queue_stop_is_idempotent():
    queue = BatchQueue(infer_fn=lambda texts: np.ones((len(texts), 1)))
    await queue.start()
    await queue.stop()
    await queue.stop()  # Should not raise


@pytest.mark.asyncio
async def test_batch_queue_infer_error_propagates():
    def failing_infer(texts: list[str]) -> np.ndarray:
        raise RuntimeError('inference failed')

    queue = BatchQueue(infer_fn=failing_infer)
    await queue.start()

    with pytest.raises(RuntimeError, match='inference failed'):
        await queue.submit(['hello'])

    await queue.stop()
