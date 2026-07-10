import os
import sys
import tempfile

import pytest

# Set up a temp database before any app imports
_db_file = os.path.join(tempfile.gettempdir(), "aimusic_test.db")
os.environ["DATABASE_URL"] = f"sqlite:///{_db_file}"
os.environ["DEEPSEEK_API_KEY"] = "test-key"

for mod in list(sys.modules.keys()):
    if mod.startswith("app."):
        del sys.modules[mod]


@pytest.fixture(autouse=True)
def setup_db():
    from app.db import init_db, drop_db
    init_db()
    yield
    drop_db()


def pytest_sessionfinish(session, exitstatus):
    try:
        os.unlink(_db_file)
    except OSError:
        pass
