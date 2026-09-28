"""alembic 0013: an existing install gains ``render_timings`` with exactly the
fresh-install schema, keeps its data, and re-running is a no-op."""
import os
import sqlite3

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _config(db_path):
    from alembic.config import Config

    cfg = Config(os.path.join(_ROOT, "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    return cfg


def _shape(conn, table):
    cols = [(r[1], r[2].upper(), r[3], r[4], r[5]) for r in conn.execute(f"PRAGMA table_info({table})")]
    index = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='index' AND name='idx_render_timings_bucket'"
    ).fetchone()
    return cols, " ".join((index[0] if index else "").split())


def _v0012_db(path):
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE alembic_version (version_num VARCHAR(64) NOT NULL)")
        conn.execute("INSERT INTO alembic_version VALUES ('0012_call_sessions')")
        conn.execute("CREATE TABLE voice_profiles (id TEXT PRIMARY KEY, name TEXT)")
        conn.execute("INSERT INTO voice_profiles VALUES ('keep', 'Mine')")


def test_upgrade_from_0012_adds_the_table_and_keeps_data(tmp_path, monkeypatch):
    from alembic import command
    from core.db import _BASE_SCHEMA

    db = tmp_path / "old.db"
    _v0012_db(db)
    monkeypatch.setenv("OMNIVOICE_DB_PATH", str(db))
    command.upgrade(_config(db), "0013_render_timings")
    command.upgrade(_config(db), "0013_render_timings")  # idempotent
    canon = sqlite3.connect(":memory:")
    canon.executescript(_BASE_SCHEMA)
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT * FROM voice_profiles").fetchall() == [("keep", "Mine")]
        assert conn.execute("SELECT version_num FROM alembic_version").fetchone()[0] == \
            "0013_render_timings"
        assert _shape(conn, "render_timings") == _shape(canon, "render_timings")
        conn.execute(
            "INSERT INTO render_timings (engine, device, num_step, text_chars, audio_seconds, "
            "wall_seconds, speed, created_at) VALUES ('omnivoice', 'mps', NULL, 10, 1.0, 2.0, NULL, 0)"
        )


def test_fresh_install_schema_is_a_no_op_for_the_migration(tmp_path, monkeypatch):
    """_BASE_SCHEMA already created the table: upgrading must not fail or alter it."""
    from alembic import command
    from core.db import _BASE_SCHEMA

    db = tmp_path / "fresh.db"
    with sqlite3.connect(db) as conn:
        conn.executescript(_BASE_SCHEMA)
        before = _shape(conn, "render_timings")
        conn.execute("CREATE TABLE alembic_version (version_num VARCHAR(64) NOT NULL)")
        conn.execute("INSERT INTO alembic_version VALUES ('0012_call_sessions')")
    monkeypatch.setenv("OMNIVOICE_DB_PATH", str(db))
    command.upgrade(_config(db), "0013_render_timings")
    with sqlite3.connect(db) as conn:
        assert _shape(conn, "render_timings") == before


def test_downgrade_removes_only_the_new_table(tmp_path, monkeypatch):
    from alembic import command

    db = tmp_path / "down.db"
    _v0012_db(db)
    monkeypatch.setenv("OMNIVOICE_DB_PATH", str(db))
    command.upgrade(_config(db), "0013_render_timings")
    command.downgrade(_config(db), "0012_call_sessions")
    with sqlite3.connect(db) as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "render_timings" not in tables and "voice_profiles" in tables


@pytest.mark.parametrize("missing", ["render_timings"])
def test_startup_self_heal_creates_the_table(tmp_path, monkeypatch, missing):
    """A DB that missed init gets the table from ensure_schema (the #710 path)."""
    import core.db as core_db

    db = tmp_path / "heal.db"
    sqlite3.connect(db).close()
    monkeypatch.setattr(core_db, "get_db", lambda: sqlite3.connect(str(db)))
    core_db.ensure_schema()
    with sqlite3.connect(db) as conn:
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (missing,)
        ).fetchone()


def test_an_early_render_timings_table_gains_the_new_columns(tmp_path, monkeypatch):
    """Dev databases created from an earlier cut of 0013 (before ref_seconds and
    cold) converge on the current table through the additive reconcile."""
    import core.db as core_db

    db = tmp_path / "early.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE render_timings (id INTEGER PRIMARY KEY, engine TEXT NOT NULL, "
            "device TEXT NOT NULL, num_step INTEGER, text_chars INTEGER NOT NULL, "
            "audio_seconds REAL NOT NULL, wall_seconds REAL NOT NULL, speed REAL, "
            "created_at REAL NOT NULL)")
        conn.execute("INSERT INTO render_timings (engine, device, text_chars, audio_seconds, "
                     "wall_seconds, created_at) VALUES ('omnivoice', 'mps', 10, 1.0, 2.0, 0)")
    monkeypatch.setattr(core_db, "get_db", lambda: sqlite3.connect(str(db)))
    core_db.ensure_schema()
    with sqlite3.connect(db) as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(render_timings)")}
        row = conn.execute("SELECT ref_seconds, cold FROM render_timings").fetchone()
    assert {"ref_seconds", "cold"} <= cols
    assert row == (None, None)
