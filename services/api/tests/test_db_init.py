"""The database directory must exist before SQLite is asked to open a file.

A fresh clone starts with no `data/` directory, and `sqlite3` does not create
one: it raises "unable to open database file" and the app fails during startup
lifespan, before a single request is served. That is the first error anyone hits
after cloning, so it gets a test.
"""
from __future__ import annotations

from app.db import ensure_sqlite_dir


def test_creates_the_missing_directory(tmp_path):
    target = tmp_path / "nested" / "app.db"
    created = ensure_sqlite_dir(f"sqlite:///{target.as_posix()}")
    assert created == target.parent
    assert target.parent.is_dir()


def test_is_idempotent(tmp_path):
    url = f"sqlite:///{(tmp_path / 'data' / 'app.db').as_posix()}"
    assert ensure_sqlite_dir(url) == ensure_sqlite_dir(url)


def test_ignores_an_in_memory_database():
    assert ensure_sqlite_dir("sqlite:///:memory:") is None


def test_ignores_a_non_sqlite_url():
    assert ensure_sqlite_dir("postgresql://user@host/db") is None


def test_strips_query_parameters(tmp_path):
    url = f"sqlite:///{(tmp_path / 'q' / 'app.db').as_posix()}?check_same_thread=false"
    assert ensure_sqlite_dir(url) == tmp_path / "q"
