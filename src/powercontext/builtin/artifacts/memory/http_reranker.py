# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HTTP-backed, globally ordered Memory reranking."""

from __future__ import annotations

from typing import Annotated

import httpx
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import MemoryRerankDecision
from powercontext.builtin.inference import InferenceUsage


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _Score(_StrictModel):
    rank: Annotated[StrictInt, Field(ge=1)]
    score: Annotated[float, Field(allow_inf_nan=False)]

    @field_validator("score", mode="before")
    @classmethod
    def require_numeric_score(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("score must be numeric")
        return value


class _ScoreResponse(_StrictModel):
    scores: tuple[_Score, ...]


class HTTPMemoryReranker:
    """Score bounded candidate batches and select from their global ordering."""

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = 32,
        policy_id: str = "http-cross-encoder-v1",
    ) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ValueError("endpoint must be a non-empty string")
        if not isinstance(policy_id, str) or not policy_id.strip():
            raise ValueError("policy_id must be a non-empty string")
        self.endpoint = endpoint
        self.client = client
        self.batch_size = batch_size
        self.policy_id = policy_id

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
        /,
    ) -> MemoryRerankDecision:
        """Return original one-based ranks ordered by service score."""

        candidate_count = len(candidates)
        if candidate_count < 1:
            raise ValueError("rerank requires at least one candidate")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1 or limit > candidate_count:
            raise ValueError("limit must be between one and the candidate count")

        scored: list[tuple[float, int]] = []
        request_count = 0
        ranked = tuple(enumerate(candidates, start=1))
        for offset in range(0, candidate_count, self.batch_size):
            batch = ranked[offset : offset + self.batch_size]
            response = await self.client.post(
                self.endpoint,
                json={
                    "query": query,
                    "candidates": [{"rank": rank, "text": candidate.text} for rank, candidate in batch],
                },
            )
            request_count += 1
            response.raise_for_status()
            try:
                payload = _ScoreResponse.model_validate(response.json())
            except (TypeError, ValueError) as error:
                raise ValueError("reranker returned an invalid response") from error

            expected = {rank for rank, _candidate in batch}
            returned = [item.rank for item in payload.scores]
            if len(returned) != len(expected) or set(returned) != expected:
                raise ValueError("reranker response must cover every batch rank exactly once")
            scored.extend((item.score, item.rank) for item in payload.scores)

        scored.sort(key=lambda item: (-item[0], item[1]))
        return MemoryRerankDecision(
            selected_ranks=tuple(rank for _score, rank in scored[:limit]),
            usage=InferenceUsage(requests=request_count),
        )


__all__ = ["HTTPMemoryReranker"]
