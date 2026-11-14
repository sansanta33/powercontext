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

"""Portable full and prefix-delta context packs."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from dataclasses import dataclass
from typing import Any

_PACK_SCHEMA = "powercontext.context-pack.v1"
_DELTA_SCHEMA = "powercontext.context-pack-delta.v1"


class ContextPackError(ValueError):
    """A context pack wire value or reconstruction is invalid."""


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(items: tuple[bytes, ...]) -> str:
    unsigned = {"schema": _PACK_SCHEMA, "items": [_encode(item) for item in items]}
    return f"sha256:{hashlib.sha256(_json_bytes(unsigned)).hexdigest()}"


def _encode(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _decode(value: object) -> bytes:
    if not isinstance(value, str):
        raise ContextPackError("pack items must be Base64 strings")
    try:
        decoded = base64.b64decode(value, validate=True)
        decoded.decode("utf-8", errors="strict")
    except (binascii.Error, UnicodeDecodeError) as error:
        raise ContextPackError("pack items must contain Base64-encoded UTF-8") from error
    return decoded


def _load_object(data: bytes) -> dict[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ContextPackError("wire objects must not contain duplicate fields")
            result[key] = value
        return result

    try:
        value = json.loads(data.decode("utf-8", errors="strict"), object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContextPackError("wire value must be valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ContextPackError("wire value must be a JSON object")
    return value


def _validate_items(items: object) -> tuple[bytes, ...]:
    if not isinstance(items, (tuple, list)):
        raise ContextPackError("items must be an array")
    validated: list[bytes] = []
    for item in items:
        if not isinstance(item, bytes):
            raise ContextPackError("in-memory items must be bytes")
        try:
            item.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise ContextPackError("items must contain valid UTF-8") from error
        validated.append(item)
    return tuple(validated)


def _pack_payload(items: tuple[bytes, ...], digest: str, byte_count: int) -> dict[str, object]:
    return {
        "schema": _PACK_SCHEMA,
        "items": [_encode(item) for item in items],
        "digest": digest,
        "byte_count": byte_count,
    }


def _pack_byte_count(items: tuple[bytes, ...], digest: str) -> int:
    count = 0
    while True:
        actual = len(_json_bytes(_pack_payload(items, digest, count)))
        if actual == count:
            return actual
        count = actual


@dataclass(frozen=True, slots=True)
class ContextPack:
    """A complete, independently verifiable sequence of UTF-8 byte chunks."""

    items: tuple[bytes, ...]
    digest: str
    byte_count: int

    @classmethod
    def build(cls, items: tuple[bytes, ...]) -> ContextPack:
        validated = _validate_items(items)
        digest = _digest(validated)
        return cls(validated, digest, _pack_byte_count(validated, digest))

    @classmethod
    def from_content(cls, content: str, *, chunk_bytes: int = 512) -> ContextPack:
        if not isinstance(content, str):
            raise ContextPackError("content must be a string")
        if not isinstance(chunk_bytes, int) or isinstance(chunk_bytes, bool) or chunk_bytes <= 0:
            raise ContextPackError("chunk_bytes must be a positive integer")
        if not content:
            return cls.build(())

        chunks: list[bytes] = []
        current = bytearray()
        for character in content:
            encoded = character.encode("utf-8")
            if len(encoded) > chunk_bytes:
                raise ContextPackError("chunk limit cannot contain one UTF-8 character")
            if current and len(current) + len(encoded) > chunk_bytes:
                chunks.append(bytes(current))
                current.clear()
            current.extend(encoded)
        if current:
            chunks.append(bytes(current))
        return cls.build(tuple(chunks))

    def to_bytes(self) -> bytes:
        expected = ContextPack.build(self.items)
        if self.digest != expected.digest or self.byte_count != expected.byte_count:
            raise ContextPackError("pack digest or byte count is invalid")
        return _json_bytes(_pack_payload(self.items, self.digest, self.byte_count))

    @classmethod
    def from_bytes(cls, data: bytes) -> ContextPack:
        payload = _load_object(data)
        if set(payload) != {"schema", "items", "digest", "byte_count"} or payload.get("schema") != _PACK_SCHEMA:
            raise ContextPackError("pack has invalid fields or schema")
        raw_items = payload["items"]
        if not isinstance(raw_items, list):
            raise ContextPackError("pack items must be an array")
        items = tuple(_decode(item) for item in raw_items)
        expected = cls.build(items)
        byte_count = payload["byte_count"]
        if type(byte_count) is not int or byte_count != len(data) or byte_count != expected.byte_count:
            raise ContextPackError("pack byte count is invalid")
        if payload["digest"] != expected.digest:
            raise ContextPackError("pack digest is invalid")
        return expected


@dataclass(frozen=True, slots=True)
class ContextPackDelta:
    """Reuse a base pack prefix and append the target suffix."""

    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...]

    @property
    def byte_count(self) -> int:
        return len(self.to_bytes())

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack) -> ContextPackDelta:
        prefix = 0
        for base_item, target_item in zip(base.items, target.items, strict=False):
            if base_item != target_item:
                break
            prefix += 1
        return cls(base.digest, target.digest, prefix, target.items[prefix:])

    def to_bytes(self) -> bytes:
        if type(self.prefix_items) is not int or self.prefix_items < 0:
            raise ContextPackError("delta prefix must be a non-negative integer")
        appended = _validate_items(self.appended)
        if not _valid_digest(self.base_digest) or not _valid_digest(self.target_digest):
            raise ContextPackError("delta digests are invalid")
        return _json_bytes(
            {
                "schema": _DELTA_SCHEMA,
                "base_digest": self.base_digest,
                "target_digest": self.target_digest,
                "prefix_items": self.prefix_items,
                "appended": [_encode(item) for item in appended],
            }
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> ContextPackDelta:
        payload = _load_object(data)
        fields = {"schema", "base_digest", "target_digest", "prefix_items", "appended"}
        if set(payload) != fields or payload.get("schema") != _DELTA_SCHEMA:
            raise ContextPackError("delta has invalid fields or schema")
        prefix = payload["prefix_items"]
        if type(prefix) is not int or prefix < 0:
            raise ContextPackError("delta prefix must be a non-negative integer")
        raw_appended = payload["appended"]
        if not isinstance(raw_appended, list):
            raise ContextPackError("delta appended must be an array")
        delta = cls(
            base_digest=payload["base_digest"],
            target_digest=payload["target_digest"],
            prefix_items=prefix,
            appended=tuple(_decode(item) for item in raw_appended),
        )
        delta.to_bytes()
        return delta


def _valid_digest(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
        return False
    return all(character in "0123456789abcdef" for character in value[7:])


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta) -> ContextPack:
    """Validate and reconstruct a target pack from a serialized-compatible delta."""

    if base.digest != delta.base_digest:
        raise ContextPackError("delta does not apply to this base pack")
    if type(delta.prefix_items) is not int or delta.prefix_items < 0 or delta.prefix_items > len(base.items):
        raise ContextPackError("delta prefix is outside the base pack")
    target = ContextPack.build((*base.items[: delta.prefix_items], *_validate_items(delta.appended)))
    if target.digest != delta.target_digest:
        raise ContextPackError("reconstructed target digest is invalid")
    return target


__all__ = [
    "ContextPack",
    "ContextPackDelta",
    "ContextPackError",
    "apply_context_pack_delta",
]
