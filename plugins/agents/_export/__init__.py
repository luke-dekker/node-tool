"""Export subsystem — requirements generator + shared helpers for
per-node export() methods.

Per DESIGN.md §G: each node's own export() emits its Python lines and
import list. Nothing in this subpackage pulls heavy third-party deps;
the exported script does, but only for the specific subset of libs the
graph actually uses. See `requirements.py` for the pure-LLM-omits-torch
invariant.
"""
