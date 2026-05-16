"""SQL identifier helpers for configured relation names."""

import re

from psycopg2 import sql

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def parse_relation_name(relation_name: str) -> tuple[str, ...]:
    value = (relation_name or "").strip()
    parts = tuple(part.strip() for part in value.split(".") if part.strip())
    if len(parts) not in (1, 2):
        raise ValueError(f"Invalid relation name: {relation_name!r}")
    for part in parts:
        if not _IDENTIFIER_RE.fullmatch(part):
            raise ValueError(f"Invalid relation identifier: {relation_name!r}")
    return parts


def relation_identifier(relation_name: str) -> sql.Identifier:
    return sql.Identifier(*parse_relation_name(relation_name))
