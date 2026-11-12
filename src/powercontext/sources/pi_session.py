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

"""Pi v3 session history parsing and incremental Source ingestion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, JsonValue, field_validator

from powercontext.sources.connectors import (
    ConnectorRunCompletion,
    ConnectorRunSession,
    ConnectorRunStatus,
)
from powercontext.sources.definitions import AdapterSourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

PI_EVENT_SOURCE_NAME = "pi-event"
PI_SESSION_CONNECTOR_NAME = "pi-session-history"


class PiSessionInput(BaseModel):
    """A complete Pi v3 JSONL session and an optional selected leaf."""

    content: bytes
    leaf_id: str | None = None

    @field_validator("leaf_id")
    @classmethod
    def validate_leaf_id(cls, value: str | None) -> str | None:
        if value is not None and not value:
            raise ValueError("leaf_id must not be empty")
        return value


class PiEventInput(BaseModel):
    """One visible event on the selected Pi session branch."""

    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


class PiEventSource(Source):
    """An immutable captured Pi event."""

    source_id: str
    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


def _load_session(value: PiSessionInput) -> tuple[str, list[dict[str, JsonValue]]]:
    try:
        text = value.content.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("Pi session must be valid UTF-8") from error

    records: list[object] = []
    try:
        for line in text.splitlines():
            if line.strip():
                records.append(json.loads(line))
    except json.JSONDecodeError as error:
        raise ValueError("Pi session must contain valid JSONL") from error
    if not records or not isinstance(records[0], dict):
        raise ValueError("Pi session header is missing")
    header = records[0]
    session_id = header.get("id")
    if header.get("type") != "session" or header.get("version") != 3 or not isinstance(session_id, str) or not session_id:
        raise ValueError("Pi session must start with a valid v3 session header")

    nodes: list[dict[str, JsonValue]] = []
    seen: set[str] = set()
    for raw in records[1:]:
        if not isinstance(raw, dict):
            raise ValueError("Pi session records must be JSON objects")
        event_id = raw.get("id")
        if not isinstance(event_id, str) or not event_id or event_id in seen:
            raise ValueError("Pi event IDs must be non-empty and unique")
        if "parentId" not in raw:
            raise ValueError("Pi events must declare parentId")
        parent_id = raw["parentId"]
        if parent_id is not None and (not isinstance(parent_id, str) or parent_id not in seen):
            raise ValueError("Pi event parent must be null or an earlier event")
        seen.add(event_id)
        nodes.append(raw)  # type: ignore[arg-type]
    return session_id, nodes


def _visible_text(record: dict[str, JsonValue]) -> tuple[str, str] | None:
    event_type = record.get("type")
    if not isinstance(event_type, str):
        raise ValueError("Pi event type must be a string")
    if event_type in {"compaction", "branch_summary"}:
        summary = record.get("summary")
        if isinstance(summary, str) and summary:
            return event_type, summary
        return None
    if event_type == "custom_message":
        body = _content_blocks(record.get("content"))
        return ("custom", body) if body else None
    if event_type != "message":
        return None
    message = record.get("message")
    if not isinstance(message, dict):
        raise ValueError("message events must contain a message object")
    role = message.get("role")
    if not isinstance(role, str) or not role:
        raise ValueError("message role must be a non-empty string")
    if role == "toolResult":
        body = _content_blocks(message.get("content"))
        tool_name = message.get("toolName")
        call_id = message.get("toolCallId")
        if not isinstance(tool_name, str) or not isinstance(call_id, str):
            raise ValueError("tool results must identify their tool and call")
        return role, f"[tool_result {tool_name} {call_id}] {body}"
    if role == "bashExecution":
        command = message.get("command")
        output = message.get("output")
        if not isinstance(command, str) or not isinstance(output, str):
            raise ValueError("bash executions must contain command and output strings")
        return role, f"$ {command}\n{output}"
    body = _content_blocks(message.get("content"))
    return (role, body) if body else None


def _content_blocks(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text" and isinstance(block.get("text"), str) and block["text"]:
            parts.append(block["text"])
        elif block_type == "toolCall":
            name, call_id = block.get("name"), block.get("id")
            if not isinstance(name, str) or not isinstance(call_id, str):
                raise ValueError("tool calls must identify their tool and call")
            arguments = json.dumps(block.get("arguments"), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            parts.append(f"[tool_call {name} {call_id}] {arguments}")
    return "\n".join(parts)


def parse_pi_session(value: PiSessionInput) -> tuple[PiEventInput, ...]:
    """Validate a Pi archive and return visible events on its selected branch."""

    session_id, nodes = _load_session(value)
    if not nodes:
        if value.leaf_id is not None:
            raise ValueError("selected Pi leaf does not exist")
        return ()
    by_id = {record["id"]: record for record in nodes}
    leaf_id = value.leaf_id if value.leaf_id is not None else nodes[-1]["id"]
    if leaf_id not in by_id:
        raise ValueError("selected Pi leaf does not exist")

    branch: list[dict[str, JsonValue]] = []
    current: str | None = leaf_id
    while current is not None:
        record = by_id[current]
        branch.append(record)
        parent = record["parentId"]
        current = parent if isinstance(parent, str) else None
    branch.reverse()

    events: list[PiEventInput] = []
    for record in branch:
        visible = _visible_text(record)
        if visible is None:
            continue
        role, body = visible
        event_type = record["type"]
        event_id = record["id"]
        parent_id = record["parentId"]
        assert isinstance(event_type, str) and isinstance(event_id, str)
        events.append(
            PiEventInput(
                session_id=session_id,
                event_id=event_id,
                parent_id=parent_id if isinstance(parent_id, str) else None,
                event_type=event_type,
                role=role,
                content=f"{role}: {body}",
                record=record,
            )
        )
    return tuple(events)


def _source_id(session_id: str, event_id: str) -> str:
    payload = json.dumps([session_id, event_id], ensure_ascii=False, separators=(",", ":")).encode()
    return f"pi:{hashlib.sha256(payload).hexdigest()}"


class _PiEventAdapter:
    input_class = PiEventInput
    name = PI_EVENT_SOURCE_NAME
    source_class = PiEventSource

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        source_id = _source_id(value.session_id, value.event_id)
        return PiEventSource(
            name=source_id,
            source_id=source_id,
            materialization=SourceMaterialization.CAPTURED,
            session_id=value.session_id,
            event_id=value.event_id,
            parent_id=value.parent_id,
            event_type=value.event_type,
            role=value.role,
            content=value.content,
            record=value.record,
        )

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return PiEventInput(
            session_id=source.session_id,
            event_id=source.event_id,
            parent_id=source.parent_id,
            event_type=source.event_type,
            role=source.role,
            content=source.content,
            record=source.record,
        )


class _PiEventTextEvidenceProjection:
    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PiEventSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PI_EVENT_SOURCE_NAME,
            source_id=source.source_id,
            content=source.content,
            metadata={
                "session_id": source.session_id,
                "event_id": source.event_id,
                "parent_id": source.parent_id,
                "event_type": source.event_type,
                "role": source.role,
            },
        )


PI_EVENT_SOURCE_DEFINITION = AdapterSourceDefinition(
    _PiEventAdapter(),
    projections=(_PiEventTextEvidenceProjection(),),
)


def _record_digest(record: dict[str, JsonValue]) -> str:
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


class PiSessionConnector:
    """Incrementally ingest visible events from one selected Pi branch."""

    name = PI_SESSION_CONNECTOR_NAME
    version = "1"
    source_definitions = frozenset({PI_EVENT_SOURCE_NAME})

    def __init__(self, path: str | Path, leaf_id: str | None = None) -> None:
        self.path = Path(path)
        self.leaf_id = leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        content = self.path.read_bytes()
        value = PiSessionInput(content=content, leaf_id=self.leaf_id)
        session_id, nodes = _load_session(value)
        events = parse_pi_session(value)
        records_by_id = {record["id"]: record for record in nodes}

        checkpoint = session.checkpoint
        if checkpoint is None:
            imported: dict[str, str] = {}
        else:
            if not isinstance(checkpoint, dict) or checkpoint.get("session_id") != session_id:
                raise ValueError("Pi checkpoint does not match this session")
            saved = checkpoint.get("events")
            if not isinstance(saved, dict) or not all(isinstance(key, str) and isinstance(val, str) for key, val in saved.items()):
                raise ValueError("Pi checkpoint is invalid")
            imported = dict(saved)

        for event_id, expected_digest in imported.items():
            record = records_by_id.get(event_id)
            if record is None or _record_digest(record) != expected_digest:
                raise ValueError("an imported Pi event was removed or rewritten")

        next_imported = dict(imported)
        for event in events:
            if event.event_id in imported:
                continue
            await session.submit(event.event_id, PI_EVENT_SOURCE_NAME, event)
            next_imported[event.event_id] = _record_digest(event.record)
        return ConnectorRunCompletion(
            status=ConnectorRunStatus.COMPLETE,
            checkpoint={"session_id": session_id, "events": next_imported},
        )


__all__ = [
    "PI_EVENT_SOURCE_DEFINITION",
    "PiEventInput",
    "PiEventSource",
    "PiSessionConnector",
    "PiSessionInput",
    "parse_pi_session",
]
