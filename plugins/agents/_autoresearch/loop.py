"""ExperimentLoop — drives the mutate → eval → keep/revert cycle.

Owned by `AgentOrchestrator.autoresearch_*` RPCs on a background thread.
Honors three stop conditions (OR'd): max trials, wall-clock budget,
loss_threshold. Writes a TSV ledger; tracks best-so-far + best graph hash.

Mutation region is the A/B cone for `group`, computed once at start. Each
trial:
  1. snapshot current graph
  2. call mutator to get ONE op
  3. apply_mutation(graph, op, allowlist, cone)
  4. run_eval(graph, registry, metric, budget_s)
  5. if verdict == keep AND score < best_so_far: record best; else revert
  6. append row to ledger
"""
from __future__ import annotations
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from plugins.agents._autoresearch.evaluator import run_eval
from plugins.agents._autoresearch.ledger import Ledger, LedgerRow
from plugins.agents._autoresearch.mutation import (
    MutationOp, apply_mutation, parse_mutation_json,
)


@dataclass
class LoopBudget:
    trials:          int = 5
    wall_clock_s:    float = 300.0
    loss_threshold:  float | None = None


@dataclass
class LoopState:
    run_id:            str
    trials_done:       int = 0
    best_score:        float = float("inf")
    best_graph_hash:   str = ""
    current_status:    str = "idle"   # idle | running | done | stopped | error
    current_op_kind:   str = ""
    history:           list[dict] = field(default_factory=list)
    error:             str = ""


class ExperimentLoop:
    """One autoresearch run. Single-thread; fire-and-poll via LoopState."""

    def __init__(
        self,
        *,
        run_id:       str,
        graph,
        registry,                  # OrchestratorRegistry
        mutator_fn:   Callable[[str], str],  # (recent_tsv_tail) -> raw LLM response
        budget:       LoopBudget,
        ledger:       Ledger,
        allowlist:    set[str] | None = None,
        cone:         list[str] | None = None,
        metric:       str = "val_loss",
        eval_budget_s: float = 60.0,
        train_start_params: dict | None = None,
    ):
        self.run_id = run_id
        self.graph = graph
        self.registry = registry
        self.mutator_fn = mutator_fn
        self.budget = budget
        self.ledger = ledger
        self.allowlist = allowlist
        self.cone = set(cone) if cone is not None else None
        self.metric = metric
        self.eval_budget_s = eval_budget_s
        self.train_start_params = train_start_params or {}

        self.state = LoopState(run_id=run_id)
        self.stop_flag = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Fire the loop on a background daemon thread. Non-blocking."""
        self._thread = threading.Thread(
            target=self._run, name=f"autoresearch-{self.run_id[:6]}", daemon=True,
        )
        self.state.current_status = "running"
        self._thread.start()

    def stop(self) -> None:
        self.stop_flag.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    # ── main loop ──────────────────────────────────────────────────────────

    def _run(self) -> None:
        start_time = time.time()
        try:
            while True:
                if self.stop_flag.is_set():
                    self.state.current_status = "stopped"
                    break
                if self.state.trials_done >= self.budget.trials:
                    self.state.current_status = "done"
                    break
                if (time.time() - start_time) >= self.budget.wall_clock_s:
                    self.state.current_status = "done"
                    break
                self._run_one_trial()
                if (self.budget.loss_threshold is not None
                        and self.state.best_score <= self.budget.loss_threshold):
                    self.state.current_status = "done"
                    break
        except Exception as exc:
            self.state.current_status = "error"
            self.state.error = str(exc)

    def _run_one_trial(self) -> None:
        idx = self.state.trials_done + 1
        pre_hash = self.graph.snapshot()
        trial_t0 = time.time()

        # 1. Mutator: LLM proposes one op (text-in, text-out via callable)
        recent = self.ledger.tail(n=5)
        try:
            raw = self.mutator_fn(recent)
            op = parse_mutation_json(raw)
        except Exception as exc:
            self._record_trial(
                idx, pre_hash, op_kind="parse_error",
                score=float("inf"), status="crash",
                wall=time.time() - trial_t0, error=str(exc),
            )
            return

        self.state.current_op_kind = op.op

        # 2. apply_mutation (guarded by allowlist + cone)
        ok, msg = apply_mutation(
            self.graph, op,
            allowlist=self.allowlist, cone=self.cone,
        )
        if not ok:
            self.graph.revert_to(pre_hash)
            self._record_trial(
                idx, pre_hash, op_kind=op.op,
                score=float("inf"), status="crash",
                wall=time.time() - trial_t0, error=f"apply rejected: {msg}",
            )
            return

        post_hash = self.graph.snapshot()

        # 3. Evaluate
        result = run_eval(
            registry=self.registry, metric=self.metric,
            budget_s=self.eval_budget_s,
            start_params=self.train_start_params,
            stop_flag=self.stop_flag,
            best_so_far=self.state.best_score,
        )

        # 4. Keep / discard
        kept = (result.status == "keep" and result.score < self.state.best_score)
        if kept:
            self.state.best_score = result.score
            self.state.best_graph_hash = post_hash
        else:
            self.graph.revert_to(pre_hash)

        self._record_trial(
            idx, post_hash, op_kind=op.op,
            score=result.score, status=result.status,
            wall=time.time() - trial_t0, error=result.error,
        )

    def _record_trial(self, idx, graph_hash, *, op_kind, score, status, wall, error):
        self.state.trials_done = idx
        row = LedgerRow(
            trial_idx=idx, graph_hash=graph_hash, op_kind=op_kind,
            score=score, status=status, wall_clock_s=wall, error=error or "",
        )
        self.ledger.append(row)
        self.state.history.append({
            "trial_idx":    idx,
            "graph_hash":   graph_hash,
            "op_kind":      op_kind,
            "score":        score,
            "status":       status,
            "wall_clock_s": wall,
        })
