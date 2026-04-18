"""Dev launcher: server.py + React dev server in one Ctrl+C-able process.

Usage:
    python launch_web.py

Ctrl+C terminates both children. If either child exits unexpectedly, the
other is killed so you never end up with a stale orphan.
"""
from __future__ import annotations
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).parent
WEB  = ROOT / "web"

# ANSI colors — [BE] blue, [FE] green, errors red. Terminals without ANSI
# just see the escape codes; harmless.
BE = "\033[94m[BE]\033[0m"
FE = "\033[92m[FE]\033[0m"
ER = "\033[91m"
RS = "\033[0m"

IS_WIN = os.name == "nt"
POPEN_KW: dict = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if IS_WIN else {}


def _pump(proc: subprocess.Popen, tag: str) -> None:
    """Stream a child's stdout into our own with a tag prefix."""
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.rstrip()
        if line:
            print(f"{tag} {line}", flush=True)


def _spawn(cmd: list[str] | str, cwd: Path, tag: str, shell: bool = False) -> subprocess.Popen:
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        shell=shell,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        **POPEN_KW,
    )
    threading.Thread(target=_pump, args=(proc, tag), daemon=True).start()
    return proc


def _terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if IS_WIN:
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            proc.terminate()
        proc.wait(timeout=4)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def main() -> int:
    if not WEB.exists():
        print(f"{ER}web/ not found at {WEB}{RS}")
        return 1
    if not (WEB / "node_modules").exists():
        print(f"{ER}web/node_modules missing — run `cd web && npm install` first{RS}")
        return 1

    print(f"{BE} starting python server.py")
    backend  = _spawn([sys.executable, "server.py"], ROOT, BE)

    # npm is a .cmd shim on Windows — use shell=True so PATH resolution works.
    print(f"{FE} starting vite dev server")
    frontend = _spawn("npm run dev", WEB, FE, shell=True)

    exit_code = 0
    try:
        while True:
            rc_b = backend.poll()
            rc_f = frontend.poll()
            if rc_b is not None:
                print(f"{ER}[BE] exited ({rc_b}), stopping frontend{RS}")
                exit_code = rc_b or 1
                break
            if rc_f is not None:
                print(f"{ER}[FE] exited ({rc_f}), stopping backend{RS}")
                exit_code = rc_f or 1
                break
            try:
                backend.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                pass
    except KeyboardInterrupt:
        print("\nshutting down...")
    finally:
        _terminate(frontend)
        _terminate(backend)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
