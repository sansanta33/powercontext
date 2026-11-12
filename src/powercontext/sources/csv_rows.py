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

"""Captured CSV rows with stable identities and text-evidence projections."""

from __future__ import annotations

import csv
import io
import json
from urllib.parse import quote

from pydantic import BaseModel, field_validator, model_validator

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.definitions import AdapterSourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

CSV_ROW_SOURCE_NAME = "csv-row"


class CSVDocumentInput(BaseModel):
    """One complete CSV document to parse into independently addressable rows."""

    document_id: str
    primary_key: str
    content: bytes

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("document_id must not be empty")
        if value != value.strip():
            raise ValueError("document_id must not contain surrounding whitespace")
        return value

    @field_validator("primary_key")
    @classmethod
    def validate_primary_key(cls, value: str) -> str:
        if not value:
            raise ValueError("primary_key must not be empty")
        return value


class CSVRowInput(BaseModel):
    """One validated row from a CSV document."""

    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]

    @model_validator(mode="after")
    def validate_row(self) -> CSVRowInput:
        if not self.document_id or self.document_id != self.document_id.strip():
            raise ValueError("document_id must be non-empty and trimmed")
        if not self.primary_key or self.primary_key not in self.columns:
            raise ValueError("primary_key must exactly match a column")
        if not self.primary_key_value.strip():
            raise ValueError("primary_key_value must not be blank")
        if len(self.columns) != len(self.values):
            raise ValueError("columns and values must have equal lengths")
        return self


class CSVRowSource(Source):
    """A captured CSV row that retains its original strings and column order."""

    source_id: str
    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


def csv_row_source_id(document_id: str, primary_key_value: str) -> str:
    """Build the stable, delimiter-safe identity for a CSV row."""

    if not document_id or not document_id.strip():
        raise ValueError("document_id must not be empty")
    if document_id != document_id.strip():
        raise ValueError("document_id must not contain surrounding whitespace")
    if not primary_key_value.strip():
        raise ValueError("primary_key_value must not be blank")
    source_id = f"csv:{quote(document_id, safe='')}:{quote(primary_key_value, safe='')}"
    if len(source_id) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"source_id must not exceed {MAX_SOURCE_ID_LENGTH} characters")
    return source_id


def parse_csv_document(value: CSVDocumentInput) -> tuple[CSVRowInput, ...]:
    """Parse and atomically validate a complete CSV document."""

    try:
        text = value.content.decode("utf-8-sig", errors="strict")
        records = list(csv.reader(io.StringIO(text, newline=""), delimiter=",", quotechar='"', strict=True))
    except (UnicodeDecodeError, csv.Error) as error:
        raise ValueError("content must be a valid UTF-8 CSV document") from error

    if not records:
        raise ValueError("CSV document must contain a header")
    columns = tuple(records[0])
    if not columns or any(not column.strip() for column in columns):
        raise ValueError("CSV header fields must not be blank")
    if len(set(columns)) != len(columns):
        raise ValueError("CSV header fields must be unique")
    if value.primary_key not in columns:
        raise ValueError("primary_key must exactly match a header field")

    primary_key_index = columns.index(value.primary_key)
    seen_keys: set[str] = set()
    rows: list[CSVRowInput] = []
    for record in records[1:]:
        if not record:
            continue
        if len(record) != len(columns):
            raise ValueError("CSV row width must match the header")
        primary_key_value = record[primary_key_index]
        if not primary_key_value.strip():
            raise ValueError("primary key values must not be blank")
        if primary_key_value in seen_keys:
            raise ValueError("primary key values must be unique within a document")
        seen_keys.add(primary_key_value)
        rows.append(
            CSVRowInput(
                document_id=value.document_id,
                primary_key=value.primary_key,
                primary_key_value=primary_key_value,
                columns=columns,
                values=tuple(record),
            )
        )
    return tuple(rows)


class _CSVRowAdapter:
    input_class = CSVRowInput
    name = CSV_ROW_SOURCE_NAME
    source_class = CSVRowSource

    async def resolve(self, value: CSVRowInput, /) -> CSVRowSource:
        source_id = csv_row_source_id(value.document_id, value.primary_key_value)
        return CSVRowSource(
            name=source_id,
            source_id=source_id,
            materialization=SourceMaterialization.CAPTURED,
            document_id=value.document_id,
            primary_key=value.primary_key,
            primary_key_value=value.primary_key_value,
            columns=value.columns,
            values=value.values,
        )

    async def read(self, source: CSVRowSource, /) -> CSVRowInput:
        return CSVRowInput(
            document_id=source.document_id,
            primary_key=source.primary_key,
            primary_key_value=source.primary_key_value,
            columns=source.columns,
            values=source.values,
        )


class _CSVRowTextEvidenceProjection:
    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = CSVRowSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: CSVRowSource, /) -> TextEvidence:
        row = dict(zip(source.columns, source.values, strict=True))
        return TextEvidence(
            source_type=CSV_ROW_SOURCE_NAME,
            source_id=source.source_id,
            content=json.dumps(row, ensure_ascii=False, separators=(",", ":")),
            metadata={
                "document_id": source.document_id,
                "primary_key": source.primary_key,
                "primary_key_value": source.primary_key_value,
                "columns": list(source.columns),
                "row": row,
            },
        )


CSV_ROW_SOURCE_DEFINITION = AdapterSourceDefinition(
    _CSVRowAdapter(),
    projections=(_CSVRowTextEvidenceProjection(),),
)


__all__ = [
    "CSV_ROW_SOURCE_DEFINITION",
    "CSVDocumentInput",
    "CSVRowInput",
    "CSVRowSource",
    "csv_row_source_id",
    "parse_csv_document",
]
