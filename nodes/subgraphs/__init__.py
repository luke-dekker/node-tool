"""Subgraph node discovery and dynamic class generation.

Walks the project's `subgraphs/` directory at import time. For each
.subgraph.json file found, generates a SubgraphNode subclass with:

  - type_name = "subgraph_<file_stem>"
  - label     = manifest.name
  - category  = "Subgraphs"
  - _setup_ports() reading the manifest's external_inputs/outputs

The generated classes are added to this module's namespace so the existing
nodes/__init__.py auto-discovery picks them up via _discover().

Adding a new subgraph: drop a .subgraph.json file in subgraphs/, restart the
app. The dynamic class is created at startup and shows up in the palette.
A future improvement could plug into the HotReloader for live reload.
"""
from __future__ import annotations
from pathlib import Path
from core.subgraph import SubgraphFile, SUBGRAPHS_DIR
from nodes.subgraphs._base import SubgraphNode


def _make_subgraph_class(subgraph_file: SubgraphFile, type_name: str) -> type:
    """Create a SubgraphNode subclass bound to one specific .subgraph.json file."""
    cls_name = "Subgraph_" + type_name.replace(" ", "_").replace("-", "_")
    return type(
        cls_name,
        (SubgraphNode,),
        {
            "type_name":      type_name,
            "label":          subgraph_file.name,
            "category":       "Subgraphs",
            "subcategory":    "",
            "description":    subgraph_file.description or f"Saved subgraph: {subgraph_file.name}",
            "_subgraph_file": subgraph_file,
        },
    )


def _discover_subgraphs() -> dict[str, type]:
    """Scan SUBGRAPHS_DIR for .subgraph.json files and build classes for each."""
    SUBGRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    classes: dict[str, type] = {}
    for path in sorted(SUBGRAPHS_DIR.glob("*.subgraph.json")):
        try:
            sf = SubgraphFile.load(path)
        except Exception as exc:
            print(f"[subgraphs] failed to load {path.name}: {exc}")
            continue
        type_name = f"subgraph_{path.stem.removesuffix('.subgraph')}"
        cls = _make_subgraph_class(sf, type_name)
        classes[type_name] = cls
    return classes


# At import time: scan and inject generated classes into module globals so
# nodes/__init__.py's _discover(subgraphs) picks them up.
_GENERATED = _discover_subgraphs()
globals().update(_GENERATED)


def reload_subgraphs() -> list[str]:
    """Re-scan the subgraphs directory and refresh module globals.

    Returns the list of type_names now available. Caller is responsible for
    re-running the NODE_REGISTRY discovery (typically via nodes._discover).
    """
    global _GENERATED
    # Drop old generated classes from module namespace
    for old_name in list(_GENERATED):
        cls = _GENERATED[old_name]
        globals().pop(cls.__name__, None)
    _GENERATED = _discover_subgraphs()
    globals().update(_GENERATED)
    return list(_GENERATED.keys())
