"""Agno Agent node."""
from __future__ import annotations
from typing import Any
from core.node import BaseNode, PortType


class AgnoAgentNode(BaseNode):
    type_name   = "ai_agno_agent"
    label       = "Agno Agent"
    category    = "AI"
    subcategory = "Agno"
    description = (
        "Call an Agno agent or team. "
        "Requires an Agno FastAPI server (github.com/agno-agi/agno)."
    )

    def _setup_ports(self) -> None:
        self.add_input("message",  PortType.STRING, default="",
                       description="Message to send to the agent")
        self.add_input("agent_id", PortType.STRING, default="my-agent",
                       description="Agent or team id from your Agno server")
        self.add_input("host",     PortType.STRING, default="http://localhost:8000",
                       description="Agno FastAPI host URL")
        self.add_input("stream",   PortType.BOOL,   default=False,
                       description="Use SSE streaming (returns concatenated response)")
        self.add_output("response",    PortType.STRING)
        self.add_output("__terminal__", PortType.STRING)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        message  = inputs.get("message")  or ""
        agent_id = inputs.get("agent_id") or "my-agent"
        host     = (inputs.get("host") or "http://localhost:8000").rstrip("/")
        use_stream = bool(inputs.get("stream", False))

        if not message:
            return {"response": "", "__terminal__": "[Agno] No message."}

        try:
            import requests

            # Determine endpoint: teams vs agents
            if "team" in agent_id:
                url = f"{host}/v1/teams/{agent_id}/runs"
            else:
                url = f"{host}/v1/agents/{agent_id}/runs"

            payload = {"message": message}

            if use_stream:
                # Collect SSE stream
                lines = []
                with requests.post(url, json=payload, stream=True, timeout=180) as r:
                    r.raise_for_status()
                    for raw in r.iter_lines():
                        if not raw:
                            continue
                        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                        if line.startswith("data:"):
                            chunk = line[5:].strip()
                            if chunk and chunk != "[DONE]":
                                try:
                                    import json
                                    obj = json.loads(chunk)
                                    content = (
                                        obj.get("content")
                                        or obj.get("message", {}).get("content")
                                        or ""
                                    )
                                    lines.append(content)
                                except Exception:
                                    lines.append(chunk)
                response = "".join(lines)
            else:
                resp = requests.post(url, json=payload, timeout=180)
                resp.raise_for_status()
                data = resp.json()
                response = (
                    data.get("content")
                    or data.get("message", {}).get("content")
                    or str(data)
                )

            log = f"[Agno] {agent_id} → {len(response)} chars"
            return {"response": response, "__terminal__": log}

        except Exception as exc:
            msg = f"[Agno] Error: {exc}"
            return {"response": "", "__terminal__": msg}

    def export(self, iv, ov):
        message  = self._val(iv, "message")
        agent_id = self._val(iv, "agent_id")
        host     = self._val(iv, "host")
        out      = ov.get("response", "_agno_response")
        lines    = [
            "import requests as _req",
            f"_agno_url = {host} + ('/v1/teams/' if 'team' in str({agent_id}) else '/v1/agents/') + str({agent_id}) + '/runs'",
            f"_agno_r = _req.post(_agno_url, json={{'message': {message}}}, timeout=180)",
            f"_agno_d = _agno_r.json()",
            f"{out} = _agno_d.get('content') or _agno_d.get('message', {{}}).get('content') or str(_agno_d)",
        ]
        return [], lines
