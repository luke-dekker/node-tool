"""Append-only results.tsv writer for autoresearch runs.

Columns match Karpathy's autoresearch ledger, with trial_idx as the row key:

    trial_idx  graph_hash  op_kind  score  status  wall_clock_s  error

`status` ∈ {keep, discard, crash}. `score` is the configured metric's value
at end-of-trial; `inf` on crash/timeout/abort. `error` is blank on success.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path


LEDGER_COLUMNS = ("trial_idx", "graph_hash", "op_kind", "score",
                  "status", "wall_clock_s", "error")


@dataclass
class LedgerRow:
    trial_idx:     int
    graph_hash:    str
    op_kind:       str
    score:         float
    status:        str            # keep | discard | crash
    wall_clock_s:  float
    error:         str = ""


@dataclass
class Ledger:
    """Append-only TSV ledger for one autoresearch run. Thread-safe via
    its own `_lock` — the evaluator thread writes as trials finish; the
    panel's status poll reads from the in-memory mirror.
    """
    path:   str
    rows:   list[LedgerRow] = field(default_factory=list)

    def __post_init__(self) -> None:
        import threading
        self._lock = threading.Lock()
        p = Path(self.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            p.write_text("\t".join(LEDGER_COLUMNS) + "\n", encoding="utf-8")

    def append(self, row: LedgerRow) -> None:
        with self._lock:
            self.rows.append(row)
            line = "\t".join(self._fmt(getattr(row, c)) for c in LEDGER_COLUMNS)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def tail(self, n: int = 5) -> str:
        """Return the last N rows as a TSV blob (header + rows)."""
        with self._lock:
            header = "\t".join(LEDGER_COLUMNS)
            rows = self.rows[-n:] if n > 0 else self.rows
            body = "\n".join(
                "\t".join(self._fmt(getattr(r, c)) for c in LEDGER_COLUMNS)
                for r in rows
            )
            return header + "\n" + body if body else header

    @staticmethod
    def _fmt(v):
        if isinstance(v, float):
            if v != v:          # NaN
                return "nan"
            if v == float("inf"):
                return "inf"
            return f"{v:.6f}"
        return str(v).replace("\t", " ").replace("\n", " ")
