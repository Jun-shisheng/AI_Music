import os
import sys
import tempfile

# Set up a temp database before any app imports
_db_file = os.path.join(tempfile.gettempdir(), "aimusic_test.db")
os.environ["DATABASE_URL"] = f"sqlite:///{_db_file}"
os.environ["DEEPSEEK_API_KEY"] = "test-key"

for mod in list(sys.modules.keys()):
    if mod.startswith("app."):
        del sys.modules[mod]


def pytest_sessionfinish(session, exitstatus):
    try:
        os.unlink(_db_file)
    except OSError:
        pass
