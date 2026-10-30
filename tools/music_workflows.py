"""Local entry point for fixed-singer and reference-arrangement workflows.

This dispatcher uses argument arrays, never shell-built commands. It records
execution outcomes separately from a listener's acceptance and serializes both
workflows so they cannot compete for the same GPU through this entry point.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
LOCAL_TIME = timezone(timedelta(hours=8))
LOG_ROOT = ROOT / "logs" / "music_workflows"
PYTHONS = (
    ROOT / "tools" / "vocal_repair" / ".venv" / "Scripts" / "python.exe",
    ROOT / "vendor" / "YuE2" / ".venv" / "Scripts" / "python.exe",
)


def now() -> str:
    return datetime.now(LOCAL_TIME).isoformat(timespec="seconds")


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def runtime() -> str:
    override = os.environ.get("AI_MUSIC_WORKFLOW_PYTHON")
    if override:
        path = Path(override)
        if not path.is_file():
            raise FileNotFoundError(f"AI_MUSIC_WORKFLOW_PYTHON does not exist: {path}")
        return str(path.resolve())
    return str(next((path for path in PYTHONS if path.is_file()), Path(sys.executable)))


@contextmanager
def workflow_lock():
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = LOG_ROOT / "workflow.lock"
    handle = lock_path.open("a+b")
    try:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("Another music workflow is running. Check 'status' and retry after it exits.") from error
        yield
    finally:
        handle.close()  # The OS releases the lock, including after process death.


def doctor() -> int:
    required = {
        "workflow_python": Path(runtime()),
        "yue2_python": PYTHONS[1],
        "yue2_model": ROOT / "models" / "yue2" / "YuE2-3B",
        "yue2_vae": ROOT / "models" / "yue2" / "YuE2-Vae",
        "voice_anchor": ROOT / "_archive" / "outputs_history" / "2026-10-16_epic_hook" / "selected" / "04_iron_rain_raw.wav",
        "arrangement_reference": ROOT / "human_refs" / "refs" / "hoyo-mix_wildfire.m4a",
    }
    result = {
        "checked_at": now(),
        "paths": {key: {"path": str(path), "exists": path.exists()} for key, path in required.items()},
        "executables": {name: shutil.which(name) for name in ("ffmpeg", "ffprobe", "nvidia-smi")},
        "note": "Model-specific readiness is checked by each workflow. No synthesis or quality acceptance occurs in this check.",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if all(path.exists() for path in required.values()) and shutil.which("ffmpeg") else 1


def status() -> int:
    manifests = sorted(LOG_ROOT.glob("*/run.json"), reverse=True)[:12]
    if not manifests:
        print("No workflow runs recorded yet.")
    for manifest in manifests:
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            print(f"{data['started_at']}  {data['workflow']:9} {data['status']:19} {manifest}")
        except (OSError, ValueError, KeyError):
            print(f"Unreadable run record: {manifest}")
    return 0


def stop_owned_process_tree(child: subprocess.Popen) -> None:
    """Stop this dispatcher's child and the model workers it started."""
    if child.poll() is not None:
        return
    if os.name == "nt":
        # PID comes directly from our own Popen, never from a broad process-name match.
        subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
    else:
        import signal
        os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=15)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            os.killpg(child.pid, signal.SIGKILL)
        else:
            child.kill()
        child.wait(timeout=15)


def dispatch(workflow: str, arguments: list[str]) -> int:
    module = {"voice": "tools.voice_studio", "reference": "tools.reference_studio"}[workflow]
    command = [runtime(), "-u", "-m", module, *arguments]
    # Help is read-only and does not take a GPU lock or create a run record.
    if not arguments or arguments == ["--help"] or arguments == ["-h"]:
        return subprocess.call(command, cwd=ROOT)
    with workflow_lock():
        run_id = datetime.now(LOCAL_TIME).strftime("%Y%m%d-%H%M%S") + "_" + workflow + "_" + uuid.uuid4().hex[:6]
        run_dir = LOG_ROOT / run_id
        run_dir.mkdir(parents=True)
        record = {
            "run_id": run_id, "workflow": workflow, "started_at": now(),
            "status": "running", "command": command,
            "listener_acceptance": "pending", "console_log": str(run_dir / "console.log"),
        }
        write_json(run_dir / "run.json", record)
        child = None
        try:
            env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
            with (run_dir / "console.log").open("w", encoding="utf-8") as log:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                         start_new_session=os.name != "nt")
                record["pid"] = child.pid
                write_json(run_dir / "run.json", record)
                assert child.stdout is not None
                for line in child.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                code = child.wait()
            record.update(status="command_completed" if code == 0 else "failed", exit_code=code)
            return code
        except KeyboardInterrupt:
            if child is not None and child.poll() is None:
                stop_owned_process_tree(child)
            record.update(status="interrupted", exit_code=130)
            return 130
        except Exception as error:
            if child is not None and child.poll() is None:
                stop_owned_process_tree(child)
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            record["ended_at"] = now()
            write_json(run_dir / "run.json", record)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Fixed singer + reference arrangement workflows")
    parser.add_argument("workflow", choices=("doctor", "status", "voice", "reference", "review", "package", "qa"))
    parser.add_argument("arguments", nargs=argparse.REMAINDER, help="Arguments forwarded to the selected workflow")
    args = parser.parse_args()
    if args.workflow in ("doctor", "status") and args.arguments:
        parser.error("doctor/status do not accept additional arguments")
    try:
        if args.workflow in ("review", "package"):
            return subprocess.call([sys.executable, str(ROOT / "tools/workflow_delivery.py"), *args.arguments], cwd=ROOT)
        if args.workflow == "qa":
            return subprocess.call([sys.executable, str(ROOT / "tools/music_qa.py"), *args.arguments], cwd=ROOT)
        return doctor() if args.workflow == "doctor" else status() if args.workflow == "status" else dispatch(args.workflow, args.arguments)
    except (RuntimeError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
