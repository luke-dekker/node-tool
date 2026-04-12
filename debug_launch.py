"""Debug launch — prints each import step."""
import sys, time, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)

try:
    log("importing gui.app...")
    from gui.app import NodeApp
    log("gui.app imported")

    log("creating NodeApp...")
    app = NodeApp()
    log("NodeApp created, launching...")
    app.run()
except Exception as e:
    log(f"CRASH: {e}")
    traceback.print_exc()
    sys.exit(1)
