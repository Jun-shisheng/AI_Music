"""Regression: cancelling a dispatcher must stop its model grandchildren."""
import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.music_workflows import stop_owned_process_tree


@pytest.mark.skipif(os.name != "nt", reason="Windows taskkill process-tree regression")
def test_owned_grandchild_is_stopped(tmp_path):
    helper = tmp_path / "parent.py"
    pid_file = tmp_path / "grandchild.pid"
    helper.write_text(
        "import subprocess,sys,time\nfrom pathlib import Path\n"
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)'])\n"
        "Path(sys.argv[1]).write_text(str(child.pid))\ntime.sleep(120)\n", encoding="utf-8")
    child = subprocess.Popen([sys.executable, str(helper), str(pid_file)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 10
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(.02)
        assert pid_file.exists(), "Test helper did not start its child"
        grandchild_pid = int(pid_file.read_text())
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000 | 0x00100000, False, grandchild_pid)
        assert handle
        try:
            stop_owned_process_tree(child)
            code = wintypes.DWORD()
            assert kernel.GetExitCodeProcess(handle, ctypes.byref(code))
            assert code.value != 259, "Grandchild survived dispatch cancellation"
            assert child.poll() is not None
        finally:
            kernel.CloseHandle(handle)
    finally:
        if child.poll() is None:
            stop_owned_process_tree(child)
