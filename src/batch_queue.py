import asyncio
import time
from collections.abc import Callable

import numpy as np

from src.config import MAX_BATCH_SIZE, MAX_WAIT_MS


class BatchQueue:
    def __init__(self, infer_fn: Callable[[list[str]], np.ndarray]):
        self._infer_fn = infer_fn
        self._queue: asyncio.Queue[tuple[list[str], asyncio.Future]] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None

    async def start(self):
        self._worker_task = asyncio.create_task(self._worker())

    async def stop(self):
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    async def submit(self, texts: list[str]) -> list[list[float]]:
        future = asyncio.get_running_loop().create_future()
        await self._queue.put((texts, future))
        return await future

    async def _worker(self):
        pending: tuple[list[str], asyncio.Future] | None = None
        while True:
            batch_texts: list[str] = []
            futures: list[tuple[asyncio.Future, int, int]] = []

            if pending is not None:
                texts, future = pending
                pending = None
            else:
                texts, future = await self._queue.get()

            batch_texts.extend(texts)
            futures.append((future, 0, len(texts)))

            deadline = time.monotonic() + MAX_WAIT_MS / 1000.0
            while len(batch_texts) < MAX_BATCH_SIZE:
                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    break
                try:
                    texts, future = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                    if len(batch_texts) + len(texts) > MAX_BATCH_SIZE:
                        # Doesn't fit — defer to next batch
                        pending = (texts, future)
                        break
                    start = len(batch_texts)
                    batch_texts.extend(texts)
                    futures.append((future, start, start + len(texts)))
                except asyncio.TimeoutError:
                    break

            try:
                embs = await asyncio.to_thread(self._infer_fn, batch_texts)
                for future, s, e in futures:
                    future.set_result(embs[s:e].tolist())
            except Exception as exc:
                for future, _, _ in futures:
                    if not future.done():
                        future.set_exception(exc)
