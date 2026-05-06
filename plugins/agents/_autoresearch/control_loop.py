"""ControlLoop — autoresearch driver for the wired-control AutoresearchAgent.

Replaces ExperimentLoop's role for the new agent design. Where the old
loop scoped mutations by an A/B cone + class allowlist + an op grammar
(swap_node_class / add_node / set_input / ...), this loop reads explicit
`(node_id, port_name)` targets that the agent has wired its `control`
output into. Each trial:

  1. Snapshot graph
  2. Ask the LLM for a JSON dict {target_id: new_value}, given the current
     value, port type, and choices/min/max for each target
  3. Apply by writing each new value into `node.inputs[port].default_value`
     (with type coercion + range validation)
  4. Train via `train_start` over the registry (re-using the user's last
     training-panel submission, with `group` overlaid)
  5. Score via `get_training_losses`; keep iff score < best_so_far,
     otherwise revert via `graph.revert_to(snapshot_hash)`

Runs on a daemon thread; the agent node's `_best_score` / `_best_hash` /
`_history` get updated on each trial completion so the panel + the agent's
own output ports reflect live state.
"""
from __future__ import annotations
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from plugins.agents._autoresearch.evaluator import run_eval
from plugins.agents._autoresearch.ledger import Ledger, LedgerRow


_SYSTEM_PROMPT = """\
You are an autoresearch agent tuning model hyperparameters to lower a
target metric. Each call you receive a list of controllable parameters
with their current values, types, and allowed ranges/choices. You return
a JSON object naming the changes you'd like to try this trial.

Target ids use the pattern `<NodeAlias>.<port>` where NodeAlias is the
node's human-readable alias on the canvas (e.g. `Linear2`, `InputMarker1`,
`TrainMarker1`). Targets are listed in topological order — first target
is closest to the inputs, last is closest to the loss. Each target
includes a context line "<NodeType> (in: <upstream>; out: <downstream>)"
so you can tell, for example, which Linear is the first hidden layer vs
the output head when several same-type layers are controllable.

Output JSON ONLY — no prose, no markdown fences:

  {"changes": [{"target": "<NodeAlias>.<port>", "value": <new value>}, ...]}

Rules:
  - Use the EXACT target id strings shown in the prompt (case-sensitive).
  - For STRING choice targets, value MUST be one of the listed choices.
  - For INT targets, value must be an integer.
  - For FLOAT targets, value can be int or float.
  - For BOOL targets, value must be true or false.
  - Don't repeat a configuration already in the recent results.
  - Empty changes list ({"changes": []}) is a valid no-op trial.
"""


@dataclass
class ControlBudget:
    trials:         int = 8
    wall_clock_s:   float = 900.0
    loss_threshold: float | None = None


@dataclass
class ControlState:
    run_id:           str
    trials_done:      int = 0
    best_score:       float = float("inf")
    best_graph_hash:  str = ""
    current_status:   str = "idle"   # idle | running | done | stopped | error
    current_op_kind:  str = "control"
    history:          list[dict] = field(default_factory=list)
    error:            str = ""


@dataclass
class _Target:
    """One wired (node, port) under agent control."""
    node_id:   str
    port_name: str
    target_id: str            # alias-qualified id like "Linear2.out_features"
    port_type: str            # PortType string (INT/FLOAT/STRING/BOOL/...)
    choices:   list | None
    pmin:      float | None
    pmax:      float | None
    # Context for the LLM prompt — which node type this target lives on,
    # and the types of the immediate upstream / downstream nodes so the
    # LLM has a prayer of distinguishing "layer 1" from "output head"
    # when there are 10 Linears wired. Filled in by collect_targets.
    node_type:  str = ""
    upstream:   str = ""      # e.g. "Flatten" or "Linear" or "Input"
    downstream: str = ""      # e.g. "Linear" or "Output" or "(none)"


def _node_context(graph, node_id: str) -> tuple[str, str]:
    """Return (upstream_label, downstream_label) for a node — the types of
    whichever nodes feed into / out of its tensor ports. Used so the LLM
    can distinguish multiple same-type layers by their position."""
    up_types: list[str] = []
    down_types: list[str] = []
    for c in graph.connections:
        # Skip agent control wires — they're a relationship marker, not
        # data flow, and mentioning "Autoresearch Agent" as an upstream
        # would dilute the position signal the LLM is meant to use.
        if c.from_port == "control":
            continue
        if c.to_node_id == node_id:
            src = graph.nodes.get(c.from_node_id)
            if src is not None:
                up_types.append(src.label or src.type_name)
        if c.from_node_id == node_id:
            dst = graph.nodes.get(c.to_node_id)
            if dst is not None:
                down_types.append(dst.label or dst.type_name)
    up   = ",".join(sorted(set(up_types)))   or "(input)"
    down = ",".join(sorted(set(down_types))) or "(output)"
    return up, down


def collect_targets(graph, agent_node_id: str) -> list[_Target]:
    """Walk the graph's connections to find every input port wired to
    `<agent>.control`. Returns targets in topological order — first entry
    is closest to the inputs, last is closest to the loss, giving the
    LLM positional context — crucial when many same-type layers (e.g.
    10 Linears) are all agent-controllable. Target ids are built from
    the node's canvas alias (`Linear2.out_features`) so users can map
    history-log entries to the node they see on the canvas."""
    order = graph.topological_order()
    position = {nid: i for i, nid in enumerate(order)}

    raw: list[tuple[int, str, str]] = []   # (topo_pos, node_id, port_name)
    for c in graph.connections:
        if c.from_node_id != agent_node_id or c.from_port != "control":
            continue
        target_node = graph.nodes.get(c.to_node_id)
        if target_node is None or c.to_port not in target_node.inputs:
            continue
        raw.append((position.get(c.to_node_id, 10**9), c.to_node_id, c.to_port))

    # Sort topologically, then stably by port name for deterministic ids.
    raw.sort(key=lambda t: (t[0], t[2]))

    out: list[_Target] = []
    for _pos, node_id, port_name in raw:
        target_node = graph.nodes[node_id]
        port = target_node.inputs[port_name]
        # Alias-qualified target id — matches the badge the user sees on
        # the canvas node, so history entries like "Linear2.out_features"
        # map trivially to a specific node. Falls back to a short id slice
        # for graphs whose aliases somehow weren't assigned.
        alias = target_node.alias or f"{target_node.type_name}_{target_node.id[:6]}"
        target_id = f"{alias}.{port_name}"
        up, down = _node_context(graph, node_id)
        out.append(_Target(
            node_id=node_id,
            port_name=port_name,
            target_id=target_id,
            port_type=str(port.port_type),
            choices=list(port.choices) if getattr(port, "choices", None) else None,
            pmin=getattr(port, "min", None),
            pmax=getattr(port, "max", None),
            node_type=target_node.label or target_node.type_name,
            upstream=up,
            downstream=down,
        ))
    return out


def _format_target_lines(graph, targets: list[_Target]) -> str:
    lines: list[str] = []
    for t in targets:
        node = graph.nodes[t.node_id]
        cur = node.inputs[t.port_name].default_value
        bits = [f"current={cur!r}", f"type={t.port_type}"]
        if t.choices:
            bits.append(f"choices={t.choices}")
        if t.pmin is not None:
            bits.append(f"min={t.pmin}")
        if t.pmax is not None:
            bits.append(f"max={t.pmax}")
        # Context line so the LLM can tell which Linear is early vs late.
        context = f"{t.node_type} (in: {t.upstream}; out: {t.downstream})"
        lines.append(f"  {t.target_id}  — {context}")
        lines.append(f"      ({', '.join(bits)})")
    return "\n".join(lines)


def _build_prompt(playbook: str, targets_block: str, recent: str,
                  best_so_far: float) -> str:
    parts = [
        "## Playbook", playbook.strip(), "",
        "## Controllable parameters", targets_block, "",
        f"## Best so far: {best_so_far if best_so_far != float('inf') else 'none'}",
    ]
    if recent.strip():
        parts += ["", "## Recent trials (TSV tail)", "```", recent, "```"]
    parts += ["", "Respond with ONE trial's worth of changes as a single JSON object."]
    return "\n".join(parts)


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines)
    return t


def parse_changes(raw: str) -> list[dict]:
    """Parse `{"changes": [{"target": ..., "value": ...}, ...]}` from an
    LLM response. Tolerant of fenced code blocks and surrounding prose."""
    text = _strip_fences(raw)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"no JSON object in {raw[:200]!r}")
    obj = json.loads(text[start:end + 1])
    changes = obj.get("changes")
    if not isinstance(changes, list):
        raise ValueError(f"expected `changes` list, got {type(changes).__name__}")
    out: list[dict] = []
    for c in changes:
        if not isinstance(c, dict) or "target" not in c or "value" not in c:
            raise ValueError(f"bad change entry: {c!r}")
        out.append({"target": str(c["target"]), "value": c["value"]})
    return out


def _coerce(value: Any, target: _Target) -> tuple[bool, Any, str]:
    """Validate + coerce `value` per the target's port type/choices/range.

    Returns (ok, coerced_value, error_message_if_not_ok).
    """
    pt = target.port_type
    if target.choices and value not in target.choices:
        return False, None, f"value {value!r} not in choices {target.choices}"
    try:
        if pt == "BOOL":
            v = bool(value)
        elif pt == "INT":
            v = int(value)
        elif pt == "FLOAT":
            v = float(value)
        elif pt == "STRING":
            v = str(value)
        else:
            v = value     # ANY / unknown — pass through
    except (TypeError, ValueError) as exc:
        return False, None, f"cast failed: {exc}"
    if target.pmin is not None and isinstance(v, (int, float)) and v < target.pmin:
        return False, None, f"{v} < min {target.pmin}"
    if target.pmax is not None and isinstance(v, (int, float)) and v > target.pmax:
        return False, None, f"{v} > max {target.pmax}"
    return True, v, ""


def apply_changes(graph, targets: list[_Target], changes: list[dict]) -> tuple[int, list[str]]:
    """Apply parsed `changes` to `graph` in place. Skips invalid entries
    with a recorded error string. Returns (n_applied, errors)."""
    by_id = {t.target_id: t for t in targets}
    applied = 0
    errors: list[str] = []
    for change in changes:
        tid = change["target"]
        target = by_id.get(tid)
        if target is None:
            errors.append(f"unknown target {tid!r}")
            continue
        ok, val, err = _coerce(change["value"], target)
        if not ok:
            errors.append(f"{tid}: {err}")
            continue
        node = graph.nodes.get(target.node_id)
        if node is None or target.port_name not in node.inputs:
            errors.append(f"{tid}: port no longer present")
            continue
        node.inputs[target.port_name].default_value = val
        applied += 1
    return applied, errors


class ControlLoop:
    """One autoresearch run driven by an AutoresearchAgentNode. Single
    daemon thread; fire-and-poll via ControlState."""

    def __init__(
        self,
        *,
        run_id:        str,
        graph,
        registry,
        agent_node,                     # AutoresearchAgentNode (live)
        targets:       list[_Target],
        llm,
        playbook:      str,
        budget:        ControlBudget,
        ledger:        Ledger,
        metric:        str = "val_loss",
        eval_budget_s: float = 60.0,
        train_start_params: dict | None = None,
        model:         str | None = None,
        temperature:   float = 0.4,
        log_fn:        Callable[[str], None] | None = None,
    ):
        self.run_id = run_id
        self.graph = graph
        self.registry = registry
        self.agent = agent_node
        self.targets = targets
        self.llm = llm
        self.playbook = playbook
        self.budget = budget
        self.log_fn = log_fn or (lambda _msg: None)
        self.ledger = ledger
        self.metric = metric
        self.eval_budget_s = eval_budget_s
        self.train_start_params = train_start_params or {}
        self.model = model
        self.temperature = temperature

        self.state = ControlState(run_id=run_id)
        self.stop_flag = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
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

    # ── main loop ──────────────────────────────────────────────────────

    def _run(self) -> None:
        from plugins.agents._llm.protocol import Message

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
        finally:
            self._sync_agent_state()

    def _run_one_trial(self) -> None:
        from plugins.agents._llm.protocol import Message

        idx = self.state.trials_done + 1
        pre_hash = self.graph.snapshot()
        trial_t0 = time.time()

        # 1. Ask the LLM for changes
        recent = self.ledger.tail(n=5)
        targets_block = _format_target_lines(self.graph, self.targets)
        prompt = _build_prompt(self.playbook, targets_block, recent,
                               self.state.best_score)
        try:
            result = self.llm.chat(
                [Message(role="system", content=_SYSTEM_PROMPT),
                 Message(role="user",   content=prompt)],
                model=self.model, temperature=self.temperature,
            )
            raw = result.message.content or ""
            changes = parse_changes(raw)
        except Exception as exc:
            self._record_trial(idx, pre_hash, op_kind="llm_error",
                               score=float("inf"), status="crash",
                               wall=time.time() - trial_t0, error=str(exc))
            return

        # Snapshot current values of all targets BEFORE applying so we can
        # render each change as `{alias}.{port}: old→new`. Reading from the
        # live graph rather than the snapshot payload keeps the types (int
        # vs string) clean for the summary.
        targets_by_id = {t.target_id: t for t in self.targets}
        old_by_id: dict[str, Any] = {}
        for tid, t in targets_by_id.items():
            node = self.graph.nodes.get(t.node_id)
            if node is not None and t.port_name in node.inputs:
                old_by_id[tid] = node.inputs[t.port_name].default_value

        def _fmt_val(v: Any) -> str:
            if isinstance(v, float):
                return f"{v:g}"      # 0.001 not 0.0010000000000000002
            return repr(v) if isinstance(v, str) else str(v)

        parts: list[str] = []
        for c in changes:
            tid = c["target"]
            old = old_by_id.get(tid, "?")
            parts.append(f"{tid}: {_fmt_val(old)}→{_fmt_val(c['value'])}")
        proposal_summary = ", ".join(parts) or "no changes (noop)"
        print(f"[autoresearch trial {idx}] LLM proposed: {proposal_summary}",
              flush=True)
        # Mirror into the panel's log buffer so the UI sees what was tried —
        # `print()` alone only reaches the process stdout.
        self.log_fn(f"[Autoresearch] trial {idx}: {proposal_summary}")
        # Carry the proposal onto the trial history so the panel can show
        # what was actually tried, not just the op_kind tag.
        self._last_proposal = proposal_summary

        # 2. Apply (in place; revert if score doesn't improve)
        applied, apply_errs = apply_changes(self.graph, self.targets, changes)
        if applied == 0 and apply_errs:
            self.graph.revert_to(pre_hash)
            self._record_trial(idx, pre_hash, op_kind="apply_rejected",
                               score=float("inf"), status="crash",
                               wall=time.time() - trial_t0,
                               error="; ".join(apply_errs[:3]))
            return

        post_hash = self.graph.snapshot()

        # 3. Evaluate via the training orchestrator
        eval_result = run_eval(
            registry=self.registry, metric=self.metric,
            budget_s=self.eval_budget_s,
            start_params=self.train_start_params,
            stop_flag=self.stop_flag,
            best_so_far=self.state.best_score,
        )

        # 4. Keep / revert
        kept = (eval_result.status == "keep"
                and eval_result.score < self.state.best_score)
        if kept:
            self.state.best_score = eval_result.score
            self.state.best_graph_hash = post_hash
        else:
            self.graph.revert_to(pre_hash)

        # op_kind is what the trial history shows — use the full proposal
        # summary (port=value pairs) so the user can read what was tried
        # at a glance.
        op_kind = getattr(self, "_last_proposal", "") or "noop"
        self._record_trial(idx, post_hash, op_kind=op_kind,
                           score=eval_result.score, status=eval_result.status,
                           wall=time.time() - trial_t0,
                           error=eval_result.error or ("; ".join(apply_errs) if apply_errs else ""))
        self._sync_agent_state()

    def _record_trial(self, idx, graph_hash, *, op_kind, score, status, wall, error):
        self.state.trials_done = idx
        self.ledger.append(LedgerRow(
            trial_idx=idx, graph_hash=graph_hash, op_kind=op_kind,
            score=score, status=status, wall_clock_s=wall, error=error or "",
        ))
        self.state.history.append({
            "trial_idx":    idx,
            "graph_hash":   graph_hash,
            "op_kind":      op_kind,
            "score":        score,
            "status":       status,
            "wall_clock_s": wall,
        })

    def _sync_agent_state(self) -> None:
        """Mirror loop state into the agent node so its output ports +
        the panel's polling RPC see live updates."""
        self.agent._best_score = self.state.best_score
        self.agent._best_hash  = self.state.best_graph_hash
        self.agent._history    = list(self.state.history)
