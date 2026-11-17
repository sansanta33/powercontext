# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Concurrent request coalescing for embedding models."""

from __future__ import annotations

import asyncio
import math
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any

from powercontext.builtin.inference.models import EmbeddingResult, EmbeddingVector, InferenceUsage
from powercontext.builtin.inference.protocols import EmbeddingModel


@dataclass(frozen=True, slots=True)
class _CompletedEmbedding:
    vector: EmbeddingVector
    batch: object
    usage: InferenceUsage


class CoalescingEmbeddingModel:
    """Combine concurrent embedding calls and cache completed vectors.

    Distinct texts arriving during the coalescing window are sent to the
    wrapped model in bounded batches.  Queued and running texts are shared by
    all callers, while cancellation remains local to the cancelling caller.
    """

    def __init__(
        self,
        model: EmbeddingModel,
        *,
        max_batch_size: int = 64,
        coalesce_ms: float = 5.0,
        cache_size: int = 1024,
    ) -> None:
        if isinstance(max_batch_size, bool) or not isinstance(max_batch_size, int) or max_batch_size <= 0:
            raise ValueError("max_batch_size must be a positive integer")
        if (
            isinstance(coalesce_ms, bool)
            or not isinstance(coalesce_ms, (int, float))
            or not math.isfinite(coalesce_ms)
            or coalesce_ms < 0
        ):
            raise ValueError("coalesce_ms must be a finite non-negative number")
        if isinstance(cache_size, bool) or not isinstance(cache_size, int) or cache_size < 0:
            raise ValueError("cache_size must be a non-negative integer")

        self._model = model
        self.profile = model.profile
        self._max_batch_size = max_batch_size
        self._coalesce_seconds = float(coalesce_ms) / 1000.0
        self._cache_size = cache_size
        self._cache: OrderedDict[str, EmbeddingVector] = OrderedDict()
        self._pending: dict[str, asyncio.Future[_CompletedEmbedding]] = {}
        self._queue: deque[str] = deque()
        self._lock = asyncio.Lock()
        self._drain_task: asyncio.Task[None] | None = None

    async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        """Return vectors in input order, sharing provider work where possible."""

        if not texts:
            return EmbeddingResult(vectors=(), usage=InferenceUsage(requests=0))

        resolved: dict[str, EmbeddingVector] = {}
        waits: dict[str, asyncio.Future[_CompletedEmbedding]] = {}
        async with self._lock:
            for text in dict.fromkeys(texts):
                cached = self._cache.get(text)
                if cached is not None:
                    self._cache.move_to_end(text)
                    resolved[text] = cached
                    continue

                future = self._pending.get(text)
                if future is None:
                    future = asyncio.get_running_loop().create_future()
                    # Consume an otherwise-unobserved exception if every waiter
                    # is cancelled. Awaiting the future still raises normally.
                    future.add_done_callback(self._consume_exception)
                    self._pending[text] = future
                    self._queue.append(text)
                waits[text] = future

            if self._queue and (self._drain_task is None or self._drain_task.done()):
                self._drain_task = asyncio.create_task(self._drain())

        completed: dict[str, _CompletedEmbedding] = {}
        if waits:
            results = await asyncio.gather(*(asyncio.shield(future) for future in waits.values()))
            completed.update(zip(waits, results, strict=True))
            resolved.update((text, result.vector) for text, result in completed.items())

        batches: dict[int, InferenceUsage] = {}
        for result in completed.values():
            batches.setdefault(id(result.batch), result.usage)
        usage = self._sum_usage(tuple(batches.values()))
        return EmbeddingResult(vectors=tuple(resolved[text] for text in texts), usage=usage)

    async def _drain(self) -> None:
        await asyncio.sleep(self._coalesce_seconds)
        while True:
            async with self._lock:
                if not self._queue:
                    self._drain_task = None
                    return
                batch_texts = tuple(self._queue.popleft() for _ in range(min(self._max_batch_size, len(self._queue))))

            batch_identity = object()
            try:
                result = await self._model.embed(batch_texts)
                if len(result.vectors) != len(batch_texts):
                    raise ValueError(
                        "embedding model returned "
                        f"{len(result.vectors)} vectors for {len(batch_texts)} texts"
                    )
            except BaseException as error:
                async with self._lock:
                    for text in batch_texts:
                        future = self._pending.pop(text)
                        if not future.done():
                            future.set_exception(error)
                continue

            async with self._lock:
                for text, vector in zip(batch_texts, result.vectors, strict=True):
                    if self._cache_size:
                        self._cache[text] = vector
                        self._cache.move_to_end(text)
                        while len(self._cache) > self._cache_size:
                            self._cache.popitem(last=False)
                    future = self._pending.pop(text)
                    if not future.done():
                        future.set_result(_CompletedEmbedding(vector, batch_identity, result.usage))

    @staticmethod
    def _consume_exception(future: asyncio.Future[Any]) -> None:
        if not future.cancelled():
            future.exception()

    @staticmethod
    def _sum_usage(usages: tuple[InferenceUsage, ...]) -> InferenceUsage:
        if not usages:
            return InferenceUsage(requests=0)

        def total(field: str) -> int | None:
            values = tuple(getattr(usage, field) for usage in usages)
            return None if any(value is None for value in values) else sum(values)  # type: ignore[arg-type]

        return InferenceUsage(
            requests=sum(usage.requests for usage in usages),
            input_tokens=total("input_tokens"),
            output_tokens=total("output_tokens"),
        )
