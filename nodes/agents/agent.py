"""AgentNode — chat with optional function-calling tool loop.

Phase A: single chat() turn with no tools.
Phase B: up to 4 bound tools + iteration loop. Standard OpenAI/Ollama
function-calling shape. Tools with side_effect=True require explicit opt-in.

Variadic tool ports are deeper foundation work (the framework today uses
fixed named ports). For Phase B v1 we expose 4 named tool slots — a Toolset
node landing later removes the cap.
"""
from __future__ import annotations
import json
from typing import Any

from core.node import BaseNode, PortType


# Hard cap on the function-calling loop. Stops runaway models from chewing
# budget if they keep emitting tool_calls instead of a final answer.
_TOOL_LOOP_HARD_CAP = 16


class AgentNode(BaseNode):
    type_name   = "ag_agent"
    label       = "Agent"
    category    = "Agents"
    subcategory = "Agent"
    description = ("Send a conversation to an LLM. If tools are bound and the "
                   "model emits tool_calls, dispatch them and loop until done.")

    def _setup_ports(self) -> None:
        self.add_input("llm", "LLM", default=None,
                       description="LLM client (from OllamaClientNode etc.)")
        self.add_input("messages", "CONVERSATION", default=None,
                       description="Input conversation")
        self.add_input("system_prompt", PortType.STRING, default="",
                       description="Optional system prompt prepended to messages")
        self.add_input("model", PortType.STRING, default="",
                       description="Override model (blank = use client's default)")
        self.add_input("temperature", PortType.FLOAT, default=0.7,
                       description="Sampling temperature")
        # Phase B: bound tools (4 slots; bundle via a Toolset node when more are needed)
        for slot in (1, 2, 3, 4):
            self.add_input(f"tool_{slot}", "TOOL", default=None,
                           description=f"Optional bound tool #{slot}")
        self.add_input("max_iterations", PortType.INT, default=5,
                       description="Cap on the function-calling loop (1 = no tools).")
        self.add_input("allow_side_effect_tools", PortType.BOOL, default=False,
                       description=("Required to invoke tools whose side_effect=True. "
                                    "See plugins/agents/DESIGN.md §B.5."))
        self.add_output("response", "CONVERSATION",
                        description="Conversation including the assistant's reply")
        self.add_output("final_message", "MESSAGE",
                        description="The assistant's final message")
        self.add_output("text", PortType.STRING,
                        description="Plain text content of the final message")
        self.add_output("tool_calls", PortType.ANY,
                        description="List of (name, args, result) for tool calls executed")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        from plugins.agents._llm.protocol import Message  # deferred

        llm = inputs.get("llm")
        if llm is None:
            raise RuntimeError("AgentNode: no LLM client connected")

        msgs_in = list(inputs.get("messages") or [])
        sys_prompt = (inputs.get("system_prompt") or "").strip()
        if sys_prompt:
            msgs_in = [Message(role="system", content=sys_prompt)] + [
                m for m in msgs_in
                if not (isinstance(m, Message) and m.role == "system")
            ]
        if not msgs_in:
            raise RuntimeError("AgentNode: no messages to send")

        model = (inputs.get("model") or "").strip() or None
        temp_in = inputs.get("temperature")
        temp = float(temp_in if temp_in is not None else 0.7)

        tools = self._collect_tools(inputs)
        max_iter_in = inputs.get("max_iterations")
        max_iter = max(1, int(max_iter_in if max_iter_in is not None else 5))
        max_iter = min(max_iter, _TOOL_LOOP_HARD_CAP)
        allow_side = bool(inputs.get("allow_side_effect_tools"))

        messages = list(msgs_in)
        tool_log: list[dict] = []
        wire_tools = [t.to_openai() for t in tools] if tools else None
        tools_by_name = {t.name: t for t in tools}

        # Iteration loop. Stops when assistant returns no tool_calls or budget hit.
        for _ in range(max_iter):
            result = llm.chat(messages, model=model, tools=wire_tools, temperature=temp)
            messages.append(result.message)
            calls = result.message.tool_calls or []
            if not calls:
                break
            if not tools:
                # Model hallucinated tool calls without any bound — stop and surface.
                break
            for call in calls:
                name, args = _parse_tool_call(call)
                tool_msg, log_entry = _dispatch_tool_call(
                    name, args, call, tools_by_name, allow_side,
                )
                tool_log.append(log_entry)
                messages.append(tool_msg)
        else:
            # Hit max_iterations without a tool-call-free assistant turn.
            messages.append(Message(
                role="assistant",
                content=f"[Agent] max_iterations ({max_iter}) reached without resolution.",
            ))

        final = messages[-1]
        return {
            "response":      messages,
            "final_message": final,
            "text":          final.content if hasattr(final, "content") else "",
            "tool_calls":    tool_log,
        }

    def export(self, iv, ov):
        return [], [f"# AgentNode export pending Phase D"]

    @staticmethod
    def _collect_tools(inputs: dict[str, Any]) -> list:
        """Pick up the connected TOOL slots in declaration order."""
        out = []
        for slot in (1, 2, 3, 4):
            t = inputs.get(f"tool_{slot}")
            if t is not None:
                out.append(t)
        return out


def _parse_tool_call(call: Any) -> tuple[str, dict]:
    """Extract (name, args_dict) from a backend tool_call entry.

    Ollama returns dict args directly; OpenAI returns a JSON-encoded string.
    """
    fn = (call.get("function") if isinstance(call, dict)
          else getattr(call, "function", {})) or {}
    if isinstance(fn, dict):
        name = str(fn.get("name", "")).strip()
        raw_args = fn.get("arguments", {})
    else:
        name = str(getattr(fn, "name", "")).strip()
        raw_args = getattr(fn, "arguments", {})

    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError:
            args = {"_raw": raw_args}
    elif isinstance(raw_args, dict):
        args = raw_args
    else:
        args = {"_raw": raw_args}
    return name, args


def _dispatch_tool_call(name: str, args: dict, call: Any,
                        tools_by_name: dict, allow_side_effect: bool):
    """Execute a single tool call, returning (tool_message, log_entry)."""
    from plugins.agents._llm.protocol import Message  # deferred

    call_id = (call.get("id") if isinstance(call, dict)
               else getattr(call, "id", None))
    log_entry = {"name": name, "args": args, "result": None, "error": None}

    tool = tools_by_name.get(name)
    if tool is None:
        result_text = f"[error] unknown tool: {name!r}"
        log_entry["error"] = result_text
    elif tool.side_effect and not allow_side_effect:
        result_text = (f"[error] tool {name!r} has side_effect=True; "
                       "set allow_side_effect_tools=True to invoke it.")
        log_entry["error"] = result_text
    else:
        try:
            result = tool.callable(**args) if tool.callable else None
            result_text = str(result) if result is not None else "ok"
            log_entry["result"] = result
        except Exception as exc:
            result_text = f"[error] {type(exc).__name__}: {exc}"
            log_entry["error"] = result_text

    return Message(
        role="tool", name=name, content=result_text, tool_call_id=call_id,
    ), log_entry
