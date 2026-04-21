"""AgentOrchestrator — high-level driver for agent runs.

Phase A surface: list local models, ping backend, list AgentNodes in the
graph, start/stop a single chat completion, get_state, drain_logs.
Phase B surface (added here): `agent_start_stream` spawns a background
thread that calls `llm.stream(...)` and pushes tokens into a per-session
buffer; `agent_drain_tokens` returns accumulated pieces + a `done` flag;
`agent_stop` cooperatively cancels the stream.

Streaming only covers the plain-text path: tokens are yielded directly from
`LLMClient.stream()`, not through `AgentNode`'s tool-calling loop. Tool-
calling runs continue to use the blocking `agent_start` RPC. This matches
DESIGN.md §A.5 — the graph-time AgentNode output is post-stream, and
streaming is purely a panel affordance.

Mirrors plugins/pytorch/training_orchestrator.py shape: one orchestrator
per app, stateful, exposes handle_rpc(method, params) for both server.py
(JSON-RPC) and gui/app.py.dispatch_rpc (in-process).
"""
from __future__ import annotations
import threading
import time
from typing import Any
from uuid import uuid4

from core.graph import Graph


class AgentOrchestrator:
    """One per app. Holds a reference to the live graph; every frontend talks
    to the same instance via handle_rpc."""

    def __init__(self, graph: Graph):
        self.graph = graph
        self._pending_logs: list[str] = []
        self._sessions: dict[str, dict] = {}   # session_id → snapshot dict
        # Streaming state (Phase B). Each dict is keyed by session_id.
        self._pending_tokens: dict[str, list[str]] = {}
        self._stream_done:    dict[str, bool]      = {}
        self._stop_flags:     dict[str, threading.Event] = {}
        self._stream_threads: dict[str, threading.Thread] = {}
        self._state_lock = threading.Lock()
        # Autoresearch state (Phase C). run_id → ExperimentLoop instance.
        self._autoresearch_runs: dict[str, Any] = {}
        # Cross-plugin RPC registry, set via attach_registry() when the
        # orchestrator is resolved from an OrchestratorRegistry. Autoresearch
        # needs it to call train_start on the pytorch plugin.
        self._registry = None

    def attach_registry(self, registry) -> None:
        """Called by OrchestratorRegistry after construction — gives us a
        back-reference for cross-plugin RPC calls (autoresearch → pytorch).
        """
        self._registry = registry

    # ── Backend enumeration ──────────────────────────────────────────────

    def list_local_models(self, params: dict | None = None) -> dict:
        """Find an OllamaClientNode in the graph (if any) and ask Ollama for
        its installed model list. Returns DynamicFormSection-shaped items.
        """
        host = self._infer_ollama_host()
        try:
            from plugins.agents._llm.ollama_client import OllamaClient  # deferred
            cli = OllamaClient(host=host)
            models = cli.list_models()
            items = [{
                "key":      m.name,
                "label":    m.name,
                "name":     m.name,
                "size_h":   m.size_h,
                "modified": m.modified,
            } for m in models]
            return {"items": items, "host": host, "ok": True, "error": ""}
        except Exception as exc:
            return {"items": [], "host": host, "ok": False, "error": str(exc)}

    def ping_backend(self, params: dict | None = None) -> dict:
        params = params or {}
        host = str(params.get("host") or self._infer_ollama_host())
        try:
            from plugins.agents._llm.ollama_client import OllamaClient
            ok = OllamaClient(host=host).ping()
            return {"ok": ok, "host": host}
        except Exception as exc:
            return {"ok": False, "host": host, "error": str(exc)}

    def list_agent_nodes(self, params: dict | None = None) -> dict:
        items = []
        for n in self.graph.nodes.values():
            if getattr(n, "type_name", "") == "ag_agent":
                items.append({"key": n.id, "label": f"Agent {n.id[:6]}"})
        return {"items": items}

    # ── Run lifecycle (Phase A: blocking single-shot chat) ──────────────

    def start(self, params: dict) -> dict:
        agent_id = str(params.get("agent_id") or "").strip()
        if not agent_id:
            for n in self.graph.nodes.values():
                if getattr(n, "type_name", "") == "ag_agent":
                    agent_id = n.id
                    break
        node = self.graph.get_node(agent_id) if agent_id else None
        if node is None or getattr(node, "type_name", "") != "ag_agent":
            return {"ok": False, "error": "No AgentNode in graph"}

        message_text = str(params.get("message") or "").strip()
        if not message_text:
            return {"ok": False, "error": "Empty message"}

        from plugins.agents._llm.protocol import Message  # deferred

        sys_text = str(
            params.get("system_prompt") or
            (node.inputs["system_prompt"].default_value
             if "system_prompt" in node.inputs else "") or ""
        )
        model = str(params.get("model") or "").strip()
        temp_default = (node.inputs["temperature"].default_value
                        if "temperature" in node.inputs else 0.7)
        temp = float(params.get("temperature") if params.get("temperature") is not None
                     else temp_default)

        llm = self._resolve_llm_for_agent(node)
        if llm is None:
            return {"ok": False,
                    "error": "No LLM client connected to AgentNode's `llm` input"}

        msgs: list[Message] = []
        if sys_text.strip():
            msgs.append(Message(role="system", content=sys_text))
        msgs.append(Message(role="user", content=message_text))

        session_id = str(uuid4())
        self._sessions[session_id] = {
            "status": "running", "started_at": time.time(),
        }
        self._pending_logs.append(
            f"[Agent] start session {session_id[:6]} -> {model or '(default)'}"
        )

        try:
            result = llm.chat(msgs, model=(model or None), temperature=temp)
        except Exception as exc:
            self._sessions[session_id].update({"status": "error", "error": str(exc)})
            self._pending_logs.append(f"[Agent] error: {exc}")
            return {"ok": False, "error": str(exc), "session_id": session_id}

        self._sessions[session_id].update({
            "status":     "done",
            "model":      result.model,
            "tokens_in":  result.tokens_in,
            "tokens_out": result.tokens_out,
            "latency_ms": result.latency_ms,
            "reply":      result.message.content,
        })
        self._pending_logs.append(
            f"[Agent] done {session_id[:6]} "
            f"({result.tokens_out}t out, {result.latency_ms:.0f}ms)"
        )
        return {
            "ok":         True,
            "session_id": session_id,
            "reply":      result.message.content,
            "tokens_in":  result.tokens_in,
            "tokens_out": result.tokens_out,
            "latency_ms": result.latency_ms,
            "model":      result.model,
        }

    def stop(self, params: dict | None = None) -> dict:
        """Cancel a streaming run if one is active for `session_id`.

        For Phase A blocking runs there's nothing to cancel mid-flight, but we
        still mark the session aborted for symmetry with Phase B streaming.
        """
        sid = str((params or {}).get("session_id") or "").strip()
        flag = self._stop_flags.get(sid)
        if flag is not None:
            flag.set()
            return {"ok": True, "status": "aborting"}
        if sid and sid in self._sessions and self._sessions[sid]["status"] == "running":
            self._sessions[sid]["status"] = "aborted"
            return {"ok": True, "status": "aborted"}
        return {"ok": True, "status": "noop"}

    def get_state(self, params: dict | None = None) -> dict:
        sid = str((params or {}).get("session_id") or "").strip()
        if sid and sid in self._sessions:
            s = self._sessions[sid]
        elif self._sessions:
            sid = max(self._sessions, key=lambda k: self._sessions[k].get("started_at", 0))
            s = self._sessions[sid]
        else:
            return {"status": "Idle", "model": "", "tokens_in": 0,
                    "tokens_out": 0, "latency_ms": 0, "reply": "", "error": ""}
        status_label = {
            "running":   "Running",   "streaming": "Streaming",
            "done":      "Done",      "error":     "Error",
            "aborted":   "Aborted",
        }.get(s.get("status", ""), s.get("status", ""))
        return {
            "status":     status_label,
            "model":      s.get("model", ""),
            "tokens_in":  int(s.get("tokens_in", 0)),
            "tokens_out": int(s.get("tokens_out", 0)),
            "latency_ms": int(s.get("latency_ms", 0)),
            "reply":      s.get("reply", ""),
            "error":      s.get("error", ""),
        }

    def drain_tokens(self, params: dict | None = None) -> dict:
        """Pop all buffered tokens for `session_id`. Returns `done=True` once
        the background thread has finished and all tokens have been drained.

        Poll pattern: the panel calls this every ~100ms while a session is
        active; each call returns only the NEW chunks since the last drain,
        never the full transcript. The caller concatenates chunks itself.
        """
        sid = str((params or {}).get("session_id") or "").strip()
        # Session unknown OR already fully drained.
        if not sid:
            # Fallback: drain the most recent streaming session if any exists.
            if self._pending_tokens:
                sid = next(reversed(self._pending_tokens))
            else:
                return {"chunks": [], "done": True, "session_id": ""}
        with self._state_lock:
            chunks = self._pending_tokens.get(sid, [])
            self._pending_tokens[sid] = []
            done = self._stream_done.get(sid, True)
        return {"chunks": chunks, "done": bool(done), "session_id": sid}

    # ── Streaming ────────────────────────────────────────────────────────

    def start_stream(self, params: dict) -> dict:
        """Start a streaming chat run on a background thread. Returns the
        session_id immediately; the caller polls `drain_tokens` to pick up
        pieces and `get_agent_state` for final metrics.

        No tool loop — the streaming path yields content tokens only. Tool-
        calling runs use the blocking `agent_start` RPC (AgentNode.execute).
        """
        agent_id = str(params.get("agent_id") or "").strip()
        if not agent_id:
            for n in self.graph.nodes.values():
                if getattr(n, "type_name", "") == "ag_agent":
                    agent_id = n.id
                    break
        node = self.graph.get_node(agent_id) if agent_id else None
        if node is None or getattr(node, "type_name", "") != "ag_agent":
            return {"ok": False, "error": "No AgentNode in graph"}

        message_text = str(params.get("message") or "").strip()
        if not message_text:
            return {"ok": False, "error": "Empty message"}

        from plugins.agents._llm.protocol import Message  # deferred

        sys_text = str(
            params.get("system_prompt") or
            (node.inputs["system_prompt"].default_value
             if "system_prompt" in node.inputs else "") or ""
        )
        model = str(params.get("model") or "").strip()
        temp_default = (node.inputs["temperature"].default_value
                        if "temperature" in node.inputs else 0.7)
        temp = float(params.get("temperature") if params.get("temperature") is not None
                     else temp_default)

        llm = self._resolve_llm_for_agent(node)
        if llm is None:
            return {"ok": False,
                    "error": "No LLM client connected to AgentNode's `llm` input"}
        if not hasattr(llm, "stream"):
            return {"ok": False,
                    "error": f"LLM backend {type(llm).__name__} does not support streaming"}

        msgs: list[Message] = []
        if sys_text.strip():
            msgs.append(Message(role="system", content=sys_text))
        msgs.append(Message(role="user", content=message_text))

        session_id = str(uuid4())
        with self._state_lock:
            self._sessions[session_id] = {
                "status": "streaming", "started_at": time.time(),
                "model": model, "reply": "",
            }
            self._pending_tokens[session_id] = []
            self._stream_done[session_id] = False
            self._stop_flags[session_id] = threading.Event()
        self._pending_logs.append(
            f"[Agent] stream start {session_id[:6]} -> {model or '(default)'}"
        )

        t = threading.Thread(
            target=self._run_stream, name=f"ag-stream-{session_id[:6]}",
            args=(session_id, llm, msgs, model, temp), daemon=True,
        )
        self._stream_threads[session_id] = t
        t.start()
        return {"ok": True, "session_id": session_id}

    def _run_stream(self, session_id: str, llm: Any, messages: list,
                    model: str, temperature: float) -> None:
        """Background worker: pushes token pieces into the session buffer.

        Catches any exception and records it on the session snapshot rather
        than letting the thread die silently.
        """
        stop_flag = self._stop_flags[session_id]
        buf_parts: list[str] = []
        t0 = time.time()
        status = "done"
        error = ""
        try:
            kwargs: dict[str, Any] = {"temperature": temperature}
            if model:
                kwargs["model"] = model
            for piece in llm.stream(messages, **kwargs):
                if stop_flag.is_set():
                    status = "aborted"
                    break
                if not piece:
                    continue
                buf_parts.append(piece)
                with self._state_lock:
                    self._pending_tokens.setdefault(session_id, []).append(piece)
        except Exception as exc:
            status = "error"
            error = str(exc)
            self._pending_logs.append(f"[Agent] stream error: {exc}")

        reply = "".join(buf_parts)
        latency_ms = (time.time() - t0) * 1000.0
        with self._state_lock:
            self._stream_done[session_id] = True
            snap = self._sessions.setdefault(session_id, {})
            snap.update({
                "status":     status,
                "reply":      reply,
                "tokens_out": snap.get("tokens_out", 0) or len(buf_parts),
                "latency_ms": int(latency_ms),
                "error":      error,
            })
        self._pending_logs.append(
            f"[Agent] stream {status} {session_id[:6]} "
            f"({len(reply)} chars, {latency_ms:.0f}ms)"
        )

    def drain_logs(self, params: dict | None = None) -> dict:
        lines = self._pending_logs
        self._pending_logs = []
        return {"lines": lines}

    # ── Autoresearch (Phase C) ─────────────────────────────────────────

    def autoresearch_start(self, params: dict) -> dict:
        """Start a mutate→eval→keep/revert loop on a daemon thread.

        Reads config from the single `ag_autoresearch` node in the graph:
          - `llm` input → LLM client
          - control wires → set of `(node_id, port_name)` targets the agent
            may tune
          - other input defaults → trials, wall-clock, metric, etc.

        Returns {ok, run_id}.
        """
        from plugins.agents._autoresearch.ledger import Ledger
        from plugins.agents._autoresearch.control_loop import (
            ControlLoop, ControlBudget, collect_targets,
        )

        params = params or {}
        agent = self._pick_node(params.get("agent_node_id"), "ag_autoresearch")
        if agent is None:
            return {"ok": False, "error":
                    "No AutoresearchAgent node in graph. Drop one onto the "
                    "canvas and wire its `control` output into the ports you "
                    "want it to tune."}
        if self._registry is None:
            return {"ok": False,
                    "error": "No OrchestratorRegistry attached — cannot reach pytorch"}

        llm = self._resolve_input(agent, "llm")
        if llm is None:
            return {"ok": False,
                    "error": "No LLM connected to AutoresearchAgent's `llm` input"}

        targets = collect_targets(self.graph, agent.id)
        if not targets:
            return {"ok": False, "error":
                    "AutoresearchAgent has no control wires. Connect its "
                    "`control` output into one or more input ports (e.g. "
                    "linear.out_features, B.lr) to define the search space."}

        # Read agent config from input defaults.
        group        = str(agent.inputs["group"].default_value or "task_1")
        playbook     = str(agent.inputs["playbook"].default_value or "")
        metric       = str(agent.inputs["metric"].default_value or "val_loss")
        trials       = int(agent.inputs["trials"].default_value or 8)
        wall_clock_s = float(agent.inputs["wall_clock_s"].default_value or 900.0)
        eval_budget  = float(agent.inputs["eval_budget_s"].default_value or 60.0)
        lt           = float(agent.inputs["loss_threshold"].default_value or 0.0)
        loss_threshold = lt if lt > 0 else None
        temperature  = float(agent.inputs["temperature"].default_value or 0.4)
        # The LLM node (OllamaClient / OpenAICompatClient / llama.cpp) owns
        # which model to call — the agent just uses whatever client is wired
        # into its `llm` input. Per-agent model override was redundant UX.
        model        = None

        # Pull the cached training-panel submission so train_start has a
        # dataset/loss/optimizer to work with each trial.
        last_params_resp = self._registry.try_dispatch(
            "get_training_last_params", {},
        )
        unhandled = getattr(self._registry, "_UNHANDLED", object())
        cached: dict | None = None
        if last_params_resp is not unhandled and isinstance(last_params_resp, dict):
            cached = last_params_resp.get("params")
        if not cached:
            return {
                "ok": False,
                "error": ("Run training once from the Training panel before "
                          "starting autoresearch — autoresearch re-uses the "
                          "last train_start config (datasets, loss, optimizer)."),
            }
        train_start_params = dict(cached)
        train_start_params["group"] = group

        run_id = str(uuid4())
        ledger_path = f"./.node-tool/autoresearch/{run_id}/results.tsv"
        ledger = Ledger(ledger_path)

        loop = ControlLoop(
            run_id=run_id, graph=self.graph, registry=self._registry,
            agent_node=agent, targets=targets, llm=llm,
            playbook=playbook,
            budget=ControlBudget(
                trials=trials, wall_clock_s=wall_clock_s,
                loss_threshold=loss_threshold,
            ),
            ledger=ledger, metric=metric, eval_budget_s=eval_budget,
            train_start_params=train_start_params,
            model=model, temperature=temperature,
        )
        self._autoresearch_runs[run_id] = loop
        loop.start()
        self._pending_logs.append(
            f"[Autoresearch] started {run_id[:6]} "
            f"(trials={trials}, metric={metric}, group={group}, "
            f"targets={len(targets)})"
        )
        return {"ok": True, "run_id": run_id}

    def autoresearch_state(self, params: dict | None = None) -> dict:
        run_id = str((params or {}).get("run_id") or "").strip()
        if not run_id and self._autoresearch_runs:
            run_id = next(reversed(self._autoresearch_runs))
        loop = self._autoresearch_runs.get(run_id)
        if loop is None:
            return {"run_id": "", "trials_done": 0, "best_score": float("inf"),
                    "best_graph_hash": "", "current_status": "idle",
                    "current_op_kind": "", "history": []}
        st = loop.state
        return {
            "run_id":           run_id,
            "trials_done":      st.trials_done,
            "best_score":       (st.best_score if st.best_score != float("inf")
                                  else None),
            "best_graph_hash":  st.best_graph_hash,
            "current_status":   st.current_status,
            "current_op_kind":  st.current_op_kind,
            "history":          list(st.history),
            "error":            st.error,
        }

    def autoresearch_stop(self, params: dict) -> dict:
        run_id = str((params or {}).get("run_id") or "").strip()
        loop = self._autoresearch_runs.get(run_id)
        if loop is None:
            return {"ok": False, "error": f"Unknown run {run_id!r}"}
        loop.stop()
        return {"ok": True, "status": "stopping"}

    # ── Uniform RPC entry point ────────────────────────────────────────

    def handle_rpc(self, method: str, params: dict | None = None) -> Any:
        params = params or {}
        handlers = {
            "agent_list_local_models":    self.list_local_models,
            "agent_ping_backend":         self.ping_backend,
            "agent_list_agent_nodes":     self.list_agent_nodes,
            "agent_start":                self.start,
            "agent_start_stream":         self.start_stream,
            "agent_stop":                 self.stop,
            "get_agent_state":            self.get_state,
            "agent_drain_tokens":         self.drain_tokens,
            "agent_drain_logs":           self.drain_logs,
            "agent_autoresearch_start":   self.autoresearch_start,
            "agent_autoresearch_state":   self.autoresearch_state,
            "agent_autoresearch_stop":    self.autoresearch_stop,
        }
        handler = handlers.get(method)
        if handler is None:
            raise ValueError(f"Unknown agent RPC method: {method}")
        return handler(params)

    # ── Private ────────────────────────────────────────────────────────

    def _infer_ollama_host(self) -> str:
        """Use the host from the first OllamaClientNode in the graph, if any."""
        for n in self.graph.nodes.values():
            if getattr(n, "type_name", "") == "ag_ollama_client":
                p = n.inputs.get("host")
                if p and p.default_value:
                    return str(p.default_value)
                break
        return "http://localhost:11434"

    def _resolve_llm_for_agent(self, agent_node) -> Any:
        """Walk back through the graph to find the LLM client feeding this
        AgentNode's `llm` input. Executes the source node to materialize the
        client. Returns None if nothing connected.
        """
        return self._resolve_input(agent_node, "llm")

    def _resolve_input(self, node, port_name: str) -> Any:
        """Resolve whatever is connected to `node.<port_name>` by executing
        the upstream node and returning the value on the connected output.
        """
        for c in self.graph.connections:
            if c.to_node_id != node.id or c.to_port != port_name:
                continue
            src = self.graph.get_node(c.from_node_id)
            if src is None:
                return None
            try:
                src_inputs = {k: src.inputs[k].default_value for k in src.inputs}
                out = src.execute(src_inputs)
                return out.get(c.from_port)
            except Exception as exc:
                self._pending_logs.append(
                    f"[Agent] {port_name} resolve failed: {exc}"
                )
                return None
        return None

    def _pick_node(self, node_id: str | None, fallback_type: str):
        """Return `graph.nodes[node_id]` if given, else the first node of
        `fallback_type`, else None.
        """
        if node_id:
            return self.graph.get_node(str(node_id))
        for n in self.graph.nodes.values():
            if getattr(n, "type_name", "") == fallback_type:
                return n
        return None


