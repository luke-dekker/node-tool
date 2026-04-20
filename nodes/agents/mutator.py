"""MutatorNode — ask the LLM to propose ONE typed mutation against the graph.

Phase C autoresearch. Reads the textual form of the mutable region (the A/B
cone for the targeted `group`), the user-supplied `playbook` (a mini prompt
about what kinds of edits to try), and the tail of the results ledger so
the LLM knows what's been tried. Emits a `{op, ...}` dict suitable for
`plugins.agents._autoresearch.mutation.apply_mutation`.

Runs inside a Graph.execute pass, reads `self._graph` to introspect state.
"""
from __future__ import annotations
import json
from typing import Any

from core.node import BaseNode, MarkerRole, PortType


_DEFAULT_PLAYBOOK = """\
Propose ONE change per iteration. Pick from:
  - swap_node_class: try a different layer / activation / optimizer
  - set_input: tweak a hyperparameter (hidden width, dropout, lr)
  - add_node / remove_node: change capacity (add a layer, drop an activation)

Favor small changes. Output JSON ONLY, no prose, no code fences.
"""


_SYSTEM_PROMPT = """\
You are an autoresearch agent mutating a deep-learning graph to lower the
validation loss. Emit EXACTLY ONE mutation op as a JSON object. Allowed ops:

  {"op": "swap_node_class", "node_id": "<short_id>", "new_class_name": "<type_name>"}
  {"op": "set_input", "node_id": "<short_id>", "port": "<name>", "value": <scalar>}
  {"op": "add_node", "class_name": "<type_name>", "connections": [["<from_id>","<from_port>","<new_in_port>"], ...]}
  {"op": "remove_node", "node_id": "<short_id>"}
  {"op": "add_connection", "from_id": "...", "from_port": "...", "to_id": "...", "to_port": "..."}

Short ids are the first 6 characters of the full uuid shown in the graph
text. The response MUST be a single JSON object, nothing else.
"""


class MutatorNode(BaseNode):
    type_name   = "ag_mutator"
    label       = "Mutator"
    category    = "Agents"
    subcategory = "Autoresearch"
    description = ("Propose one typed mutation against the graph's A/B cone "
                   "for a given `group` using an LLM.")

    def _setup_ports(self) -> None:
        self.add_input("llm", "LLM", default=None,
                       description="LLM backend (Ollama/OpenAICompat/LlamaCpp)")
        self.add_input("group", PortType.STRING, default="task_1",
                       description="A/B marker group defining the mutable region")
        self.add_input("playbook", PortType.STRING,
                       default=_DEFAULT_PLAYBOOK,
                       description="User prompt describing what edits to try")
        self.add_input("recent_results", PortType.STRING, default="",
                       description="Tail of results.tsv ledger (last few trials)")
        self.add_input("model", PortType.STRING, default="",
                       description="Model override (blank = LLM's default)")
        self.add_input("temperature", PortType.FLOAT, default=0.5,
                       description="Sampling temperature")
        self.add_output("mutation", PortType.ANY,
                        description="Typed mutation op dict")
        self.add_output("prompt", PortType.STRING,
                        description="The full prompt sent to the LLM (for debugging)")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        from plugins.agents._llm.protocol import Message       # deferred
        from plugins.agents._autoresearch.graph_textual import serialize_graph_textual
        from plugins.agents._autoresearch.mutation import parse_mutation_json

        graph = getattr(self, "_graph", None)
        if graph is None:
            raise RuntimeError(
                "MutatorNode needs a live Graph. Add it to a graph and run graph.execute()."
            )
        llm = inputs.get("llm")
        if llm is None or not hasattr(llm, "chat"):
            raise RuntimeError("MutatorNode: llm input is required")

        group = (inputs.get("group") or "task_1").strip() or "task_1"
        cone = _cone_for_group(graph, group)
        if not cone:
            raise RuntimeError(
                f"MutatorNode: no A/B cone found for group={group!r}"
            )
        textual = serialize_graph_textual(graph, region=cone, max_chars=4000)

        playbook = (inputs.get("playbook") or _DEFAULT_PLAYBOOK).strip()
        results = (inputs.get("recent_results") or "").strip()
        model = (inputs.get("model") or "").strip() or None
        temp = float(inputs.get("temperature") or 0.5)

        user_prompt = _build_user_prompt(textual, playbook, results)
        msgs = [
            Message(role="system", content=_SYSTEM_PROMPT),
            Message(role="user",   content=user_prompt),
        ]
        result = llm.chat(msgs, model=model, temperature=temp)
        raw = result.message.content or ""
        try:
            op = parse_mutation_json(raw)
        except (ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"MutatorNode: LLM response was not a parseable mutation op: {exc}\n"
                f"Got: {raw[:400]}"
            )
        return {
            "mutation": op.to_dict(),
            "prompt":   user_prompt,
        }

    def export(self, iv, ov):
        return [], [f"# MutatorNode export pending Phase D"]


# ── Helpers ────────────────────────────────────────────────────────────────

def _cone_for_group(graph, group: str) -> list[str]:
    """A/B cone for the given group. Empty list if either marker is missing."""
    a = b = None
    for n in graph.nodes_by_role(MarkerRole.INPUT):
        if _marker_group(n) == group:
            a = n
            break
    for n in graph.nodes_by_role(MarkerRole.TRAIN_TARGET):
        if _marker_group(n) == group:
            b = n
            break
    if a is None or b is None:
        return []
    return graph.subgraph_between(a.id, b.id)


def _marker_group(node) -> str:
    port = node.inputs.get("group") if hasattr(node, "inputs") else None
    return str(port.default_value or "") if port is not None else ""


def _build_user_prompt(textual: str, playbook: str, recent_results: str) -> str:
    parts = [
        "## Playbook", playbook, "",
        "## Current graph (mutable region only)",
        "```", textual, "```", "",
    ]
    if recent_results:
        parts += ["## Recent trial results (TSV tail)",
                  "```", recent_results, "```", ""]
    parts.append("Respond with ONE mutation op as a single JSON object.")
    return "\n".join(parts)
