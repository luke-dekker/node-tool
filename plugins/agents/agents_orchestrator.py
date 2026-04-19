"""AgentOrchestrator — high-level driver for agent runs.

Phase A surface: list local models, ping backend, list AgentNodes in the
graph, start/stop a single chat completion, get_state, drain_logs.
Streaming (drain_tokens) is stubbed — tokens come in Phase B. Autoresearch
RPCs come in Phase C.

Mirrors plugins/pytorch/training_orchestrator.py shape: one orchestrator
per app, stateful, exposes handle_rpc(method, params) for both server.py
(JSON-RPC) and gui/app.py.dispatch_rpc (in-process).
"""
from __future__ import annotations
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
        # Phase A is synchronous — nothing to cancel mid-flight. Mark a session
        # aborted if one is in flight (for forward-compat with Phase B streaming).
        sid = str((params or {}).get("session_id") or "").strip()
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
            "running": "Running", "done": "Done",
            "error":   "Error",   "aborted": "Aborted",
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
        """Phase A stub — token streaming lands in Phase B."""
        return {"chunks": [], "done": True}

    def drain_logs(self, params: dict | None = None) -> dict:
        lines = self._pending_logs
        self._pending_logs = []
        return {"lines": lines}

    # ── Uniform RPC entry point ────────────────────────────────────────

    def handle_rpc(self, method: str, params: dict | None = None) -> Any:
        params = params or {}
        handlers = {
            "agent_list_local_models": self.list_local_models,
            "agent_ping_backend":      self.ping_backend,
            "agent_list_agent_nodes":  self.list_agent_nodes,
            "agent_start":             self.start,
            "agent_stop":              self.stop,
            "get_agent_state":         self.get_state,
            "agent_drain_tokens":      self.drain_tokens,
            "agent_drain_logs":        self.drain_logs,
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
        for c in self.graph.connections:
            if c.to_node_id != agent_node.id or c.to_port != "llm":
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
                    f"[Agent] LLM resolve failed: {exc}"
                )
                return None
        return None
