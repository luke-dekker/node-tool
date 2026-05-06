"""Cross-plugin RPC eval loop for one autoresearch trial.

Calls the training orchestrator through the plugin OrchestratorRegistry
(not by importing plugins.pytorch), so the agents plugin stays decoupled
from torch. Poll-and-yield so the caller's thread stays responsive —
the orchestrator's autoresearch task can honor Stop between pollings.

Status mapping (Karpathy-style):
  - done     → keep if score < best_so_far else discard
  - timeout  → crash (wall-clock budget exceeded)
  - aborted  → crash (user stopped)
  - error    → crash (training raised)

Budget: the caller passes a `budget_s` float. The eval loop also honors
`stop_flag: threading.Event` for cooperative cancellation.
"""
from __future__ import annotations
import time
from dataclasses import dataclass


@dataclass
class EvalResult:
    score:        float         # final val_loss (or configured metric)
    status:       str           # keep | discard | crash
    wall_clock_s: float
    error:        str = ""


def run_eval(
    *,
    registry,                   # OrchestratorRegistry
    metric: str = "val_loss",
    budget_s: float = 300.0,
    start_params: dict | None = None,
    poll_ms: int = 250,
    stop_flag=None,             # threading.Event | None
    best_so_far: float = float("inf"),
) -> EvalResult:
    """Run one training trial, poll until terminal, return a scored result.

    `start_params` is forwarded to `train_start` verbatim (epochs, datasets,
    optimizer, loss, device — whatever the training panel expects).
    """
    t0 = time.time()
    start_params = start_params or {}
    try:
        start_resp = registry.try_dispatch("train_start", start_params)
    except Exception as exc:
        return EvalResult(score=float("inf"), status="crash",
                          wall_clock_s=0.0, error=f"train_start: {exc}")
    if start_resp is getattr(registry, "_UNHANDLED", object()):
        return EvalResult(score=float("inf"), status="crash",
                          wall_clock_s=0.0,
                          error="no training orchestrator registered")
    # train_start may refuse before actually launching (e.g. no dataset
    # path, no markers). Surface that directly — otherwise the poll loop
    # below waits out the full wall_clock budget and reports a misleading
    # "timeout" for what's really a configuration error.
    if isinstance(start_resp, dict) and start_resp.get("ok") is False:
        return EvalResult(score=float("inf"), status="crash",
                          wall_clock_s=time.time() - t0,
                          error=str(start_resp.get("error")
                                    or "train_start refused"))

    while True:
        if stop_flag is not None and stop_flag.is_set():
            _safe_stop(registry)
            return EvalResult(score=float("inf"), status="crash",
                              wall_clock_s=time.time() - t0,
                              error="aborted by user")
        elapsed = time.time() - t0
        if elapsed >= budget_s:
            _safe_stop(registry)
            return EvalResult(score=float("inf"), status="crash",
                              wall_clock_s=elapsed,
                              error=f"wall_clock timeout ({budget_s}s)")
        state = registry.try_dispatch("get_training_state", {})
        if state is getattr(registry, "_UNHANDLED", object()):
            return EvalResult(score=float("inf"), status="crash",
                              wall_clock_s=elapsed,
                              error="get_training_state not available")
        status = str((state or {}).get("status", "")).lower()
        if status in ("done", "complete", "completed"):
            break
        if status == "error":
            return EvalResult(score=float("inf"), status="crash",
                              wall_clock_s=elapsed,
                              error=str(state.get("error", "training error")))
        # Still running — sleep briefly and loop
        time.sleep(max(0.01, poll_ms / 1000.0))

    wall = time.time() - t0
    losses = registry.try_dispatch("get_training_losses", {}) or {}
    score = _extract_score(losses, metric)
    verdict = "keep" if score < best_so_far else "discard"
    return EvalResult(score=score, status=verdict, wall_clock_s=wall)


def _safe_stop(registry) -> None:
    try:
        registry.try_dispatch("train_stop", {})
    except Exception:
        pass


def _extract_score(losses: dict, metric: str) -> float:
    """Pull the final metric value from the training orchestrator's losses dict.

    Accepts several shapes to stay tolerant of TrainingOrchestrator variants:
      - {"series": {"train": [...], "val": [...]}}  ← shape `losses()` actually returns
      - {"val_loss": [...], "train_loss": [...]} → last element
      - {"final_val_loss": float}
      - {"metrics": {"val_loss": float}}
    """
    if not isinstance(losses, dict):
        return float("inf")
    # "series" shape: map 'val_loss'/'train_loss'/'accuracy' to series keys.
    series = losses.get("series")
    if isinstance(series, dict):
        key_map = {"val_loss": "val", "train_loss": "train", "accuracy": "accuracy"}
        key = key_map.get(metric, metric)
        vals = series.get(key)
        if isinstance(vals, list) and vals:
            try:
                return float(vals[-1])
            except (TypeError, ValueError):
                pass
    if metric in losses:
        v = losses[metric]
        if isinstance(v, list) and v:
            try:
                return float(v[-1])
            except (TypeError, ValueError):
                return float("inf")
        try:
            return float(v)
        except (TypeError, ValueError):
            pass
    final_key = f"final_{metric}"
    if final_key in losses:
        try:
            return float(losses[final_key])
        except (TypeError, ValueError):
            pass
    metrics = losses.get("metrics", {})
    if isinstance(metrics, dict) and metric in metrics:
        try:
            return float(metrics[metric])
        except (TypeError, ValueError):
            pass
    return float("inf")
