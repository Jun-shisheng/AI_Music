"""The worker is stopped even when the application's lifespan exits with an error."""
import pytest

from app import main
from app.services import queue


async def test_lifespan_stops_worker_on_error(monkeypatch):
    calls = []

    async def start():
        calls.append("start")

    async def stop():
        calls.append("stop")

    monkeypatch.setattr(main, "init_db", lambda: None)
    monkeypatch.setattr(queue, "start_worker", start)
    monkeypatch.setattr(queue, "stop_worker", stop)
    with pytest.raises(RuntimeError, match="lifespan error"):
        async with main.lifespan(main.app):
            raise RuntimeError("lifespan error")
    assert calls == ["start", "stop"]
