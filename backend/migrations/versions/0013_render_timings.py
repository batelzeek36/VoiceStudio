"""Render-time estimate: per-call synthesis timings

Revision ID: 0013_render_timings
Revises: 0012_call_sessions
Create Date: 2026-09-27 00:00:00.000000

Adds ``render_timings`` (docs/adr/render-time-estimate.md): one row per
completed synthesis call with the engine id, the resolved device class, the
unmasking steps (NULL for engines without a steps control), the characters
synthesized, the audio produced, the wall time and the speed. Numbers and
identifiers only: no text, voice or path. The render-time estimate fits this
machine's own speed from these rows.

Additive + idempotent (guarded by sqlite_master) like 0008 and 0012, so a
fresh-install DB where ``core/db.py::_BASE_SCHEMA`` already created the table
is a no-op. The DDL is the exact statement ``_BASE_SCHEMA`` runs, so both paths
converge on one schema.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0013_render_timings"
down_revision: Union[str, None] = "0012_call_sessions"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS render_timings (
        id INTEGER PRIMARY KEY,
        engine TEXT NOT NULL,
        device TEXT NOT NULL,
        num_step INTEGER,
        text_chars INTEGER NOT NULL,
        audio_seconds REAL NOT NULL,
        wall_seconds REAL NOT NULL,
        speed REAL,
        created_at REAL NOT NULL
    )
"""
_CREATE_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_render_timings_bucket "
    "ON render_timings(engine, device, num_step, id)"
)


def _has_table(name: str) -> bool:
    row = op.get_bind().execute(
        sa.text("SELECT name FROM sqlite_master WHERE type='table' AND name=:n"), {"n": name}
    ).fetchone()
    return row is not None


def upgrade() -> None:
    if not _has_table("render_timings"):
        op.execute(_CREATE_TABLE)
    op.execute(_CREATE_INDEX)


def downgrade() -> None:
    if _has_table("render_timings"):
        op.execute("DROP INDEX IF EXISTS idx_render_timings_bucket")
        op.drop_table("render_timings")
