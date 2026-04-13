"""Launch the NodeTool Godot frontend + Python backend together.

Usage:
    python launch_godot.py                    # auto-detect Godot
    python launch_godot.py --godot "C:/path/to/godot.exe"
    python launch_godot.py --server-only      # just the Python server
"""
import subprocess
import sys
import os
import time
import shutil
import argparse


def find_godot() -> str | None:
    """Try to find the Godot executable on the system."""
    # Check PATH
    for name in ("godot", "godot4", "Godot_v4"):
        found = shutil.which(name)
        if found:
            return found

    # Common Windows locations
    if sys.platform == "win32":
        candidates = [
            os.path.expandvars(r"%LOCALAPPDATA%\Godot"),
            os.path.expandvars(r"%PROGRAMFILES%\Godot"),
            os.path.expanduser(r"~\scoop\apps\godot\current"),
        ]
        for base in candidates:
            if os.path.isdir(base):
                for f in os.listdir(base):
                    if f.lower().startswith("godot") and f.endswith(".exe"):
                        return os.path.join(base, f)

    return None


def main():
    parser = argparse.ArgumentParser(description="Launch NodeTool")
    parser.add_argument("--godot", default=None, help="Path to Godot executable")
    parser.add_argument("--port", type=int, default=9800, help="WebSocket port")
    parser.add_argument("--server-only", action="store_true", help="Start only the Python server")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    godot_project = os.path.join(project_root, "godot")

    # Start Python WebSocket server
    print("[Launcher] Starting Python backend on port %d..." % args.port)
    server_proc = subprocess.Popen(
        [sys.executable, os.path.join(project_root, "server.py"), "--port", str(args.port)],
        cwd=project_root,
    )
    time.sleep(2)  # give server time to start

    if args.server_only:
        print("[Launcher] Server running. Press Ctrl+C to stop.")
        try:
            server_proc.wait()
        except KeyboardInterrupt:
            server_proc.terminate()
        return

    # Find and launch Godot
    godot_path = args.godot or find_godot()
    if godot_path is None:
        print("[Launcher] Could not find Godot executable.")
        print("[Launcher] Install Godot 4.4+ and either:")
        print("  - Add it to PATH")
        print("  - Pass --godot /path/to/godot.exe")
        print("[Launcher] Server is still running on ws://127.0.0.1:%d" % args.port)
        print("[Launcher] You can open the Godot project manually: %s" % godot_project)
        try:
            server_proc.wait()
        except KeyboardInterrupt:
            server_proc.terminate()
        return

    print("[Launcher] Starting Godot: %s" % godot_path)
    godot_proc = subprocess.Popen(
        [godot_path, "--path", godot_project],
        cwd=godot_project,
    )

    # Wait for either process to exit
    try:
        godot_proc.wait()
    except KeyboardInterrupt:
        pass
    finally:
        print("[Launcher] Shutting down...")
        godot_proc.terminate()
        server_proc.terminate()


if __name__ == "__main__":
    main()
