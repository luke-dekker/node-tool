# Node Tool v3 — Architecture & Developer Guide

## What it is
A visual node-based programming environment built in Python with DearPyGui. Wire nodes together on a canvas, hit Run, and values flow through the graph. Supports general scripting, data science (NumPy/Pandas/Sklearn/SciPy), PyTorch model building & training, and inline visualization.

---

## Directory Structure

```
node-tool-v3/
├── main.py                  # Entry point — instantiates NodeApp and calls run()
├── core/
│   ├── port.py              # PortType enum + coercion rules
│   ├── node.py              # BaseNode ABC — all nodes inherit this
│   ├── graph.py             # Graph: nodes dict, connections list, topological execute()
│   ├── executor.py          # Thin wrapper around Graph with last-run state
│   ├── serializer.py        # JSON save/load (Ctrl+S / Ctrl+O)
│   └── commands.py          # Command ABC + CommandStack for undo/redo
├── nodes/
│   ├── __init__.py          # NODE_REGISTRY, get_nodes_by_category(), CATEGORY_ORDER
│   ├── math_nodes.py        # 13 nodes: Add, Multiply, Sin, Clamp, MapRange...
│   ├── logic_nodes.py       # 5 nodes: Compare, And, Or, Not, Branch
│   ├── string_nodes.py      # 8 nodes: Concat, Format, Upper, Replace...
│   ├── data_nodes.py        # 9 nodes: FloatConst, IntConst, Print, ToFloat...
│   ├── numpy_nodes.py       # 33 nodes: array ops, linalg, stats, transforms
│   ├── pandas_nodes.py      # 23 nodes: CSV/numpy import, filter, groupby, XY split
│   ├── sklearn_nodes.py     # 18 nodes: scalers, models, predict, metrics
│   ├── scipy_nodes.py       # 10 nodes: stats, signal, interpolation, curve fit
│   ├── viz_nodes.py         # 10 nodes: line, scatter, heatmap, confusion matrix...
│   └── pytorch_nodes.py     # 40 nodes: layers, tensor ops, losses, optimizers, data
├── gui/
│   ├── theme.py             # All colors, pin colors, category colors, DPG themes
│   ├── app.py               # NodeApp — the entire GUI (1 class, ~900 lines)
│   ├── training_panel.py    # TrainingController — background thread training
│   └── inference_panel.py   # InferenceController — forward pass on trained model
└── tests/
    ├── test_core.py         # Graph, topological sort, cycle detection
    ├── test_nodes.py        # All 35 original nodes
    ├── test_pytorch_nodes.py # PyTorch node execution
    ├── test_serializer.py   # Save/load roundtrip
    └── test_data_nodes.py   # NumPy/Pandas/Sklearn/SciPy/Viz nodes
```

**Total: 169 built-in nodes + unlimited custom nodes, 145 tests passing.**

---

## The Three Layers

### Layer 1 — Core (zero GUI dependency)
```
PortType → Port → BaseNode → Graph → Executor
```
- `PortType` — enum of all data types (FLOAT, INT, BOOL, STRING, ANY, TENSOR, MODULE, DATALOADER, OPTIMIZER, LOSS_FN, NDARRAY, DATAFRAME, SERIES, SKLEARN_MODEL, IMAGE)
- `BaseNode` — abstract class. Every node has `inputs: dict[str, Port]`, `outputs: dict[str, Port]`, and implements `execute(inputs) -> outputs`
- `Graph.execute()` — Kahn's algorithm topological sort, then calls each node's `execute()` in dependency order, propagating values along connections. Returns `(outputs_dict, terminal_lines)`

### Layer 2 — Nodes (just subclass BaseNode)
Each node file is self-contained. No knowledge of DPG. Fully testable headlessly.
Each class defines:
1. `type_name` — unique string key in NODE_REGISTRY
2. `label` — display name
3. `category` — which palette group it appears in
4. `_setup_ports()` — declare inputs/outputs with types and defaults
5. `execute(inputs)` — the actual computation

### Layer 3 — GUI (DearPyGui)
`NodeApp` in `gui/app.py` bridges Python nodes ↔ DPG items via four mapping dicts:
```python
dpg_node_to_node_id  # DPG integer id  → node.id (string uuid)
node_id_to_dpg       # node.id         → DPG string tag
dpg_attr_to_key      # DPG attr int id → (node_id, port_name, is_input)
dpg_link_to_conn     # DPG link int id → (from_node_id, from_port, to_node_id, to_port)
```

**Critical DPG rules learned the hard way:**
- `get_selected_nodes()` / link callbacks always return **integer IDs**, never string aliases → always register both: `dict[str_tag] = x` AND `dict[dpg.get_alias_id(str_tag)] = x`
- Never call `dpg.delete_item()` inside a callback (fires mid-render) → use a flag/queue, act between frames
- Links are children of the **node_editor**, not the node → must delete links BEFORE deleting their nodes or DPG renders dangling pointers → crash
- Delink callback fires mid-render → queue it, process between frames

**Render loop:**
```python
while dpg.is_dearpygui_running():
    _process_requests()   # Ctrl+Z/Y/C/V/S — keyboard flags set mid-render
    _flush_deletions()    # delete queued nodes/links safely between frames
    _check_selection()    # poll selection, update inspector
    _poll_training()      # drain training thread events, update loss plot
    render_dearpygui_frame()
```

---

## Port Types Quick Reference

| Type | Color | Pin Shape | Used for |
|------|-------|-----------|----------|
| FLOAT | Green | ● filled | Scalars |
| INT | Blue | ▲ filled | Integers |
| BOOL | Red | ■ filled | True/False |
| STRING | Orange | ● filled | Text |
| ANY | Grey | ○ hollow | Passthrough |
| TENSOR | Orange | ○ hollow | torch.Tensor |
| MODULE | Violet | △ hollow | nn.Module |
| DATALOADER | Teal | □ hollow | DataLoader |
| OPTIMIZER | Gold | ○ hollow | optim.Optimizer |
| LOSS_FN | Rose | ○ hollow | loss function |
| NDARRAY | Sky blue | ▲ filled | np.ndarray |
| DATAFRAME | Emerald | ■ filled | pd.DataFrame |
| SERIES | Lime | ● filled | pd.Series |
| SKLEARN_MODEL | Amber | △ hollow | sklearn estimator |
| IMAGE | Hot pink | ■ hollow | RGB uint8 ndarray — renders inline |

---

## Adding a Node Without Restarting (30 seconds)

Drop a `.py` file in `nodes/custom/`. The app picks it up within one second:

```python
# nodes/custom/my_nodes.py
from core.simple_node import node

@node(label="My Node", category="Custom")
def my_node(value: float = 1.0, scale: float = 2.0) -> float:
    return value * scale

@node(label="Stats", category="Custom", outputs={"mean": float, "std": float})
def stats(array) -> dict:
    import numpy as np
    return {"mean": float(np.mean(array)), "std": float(np.std(array))}
```

Save the file → node appears in palette → done. No restart needed.

**Supported type hints:** `float` → FLOAT, `int` → INT, `bool` → BOOL, `str` → STRING, no hint → ANY

**Multiple outputs:** use `outputs={"name": type, ...}` and return a dict.

---

## How to Add a Node the Full Way (5 minutes)

### 1. Add to the right file (or create a new one)

```python
# nodes/my_nodes.py
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.node import BaseNode
from core.port import PortType

class MyNode(BaseNode):
    type_name   = "my_node"          # unique key — no spaces
    label       = "My Node"          # shown on canvas
    category    = "MyCategory"       # palette group
    description = "Does something."

    def _setup_ports(self):
        self.add_input("value",  PortType.FLOAT, default=1.0)
        self.add_input("factor", PortType.INT,   default=2)
        self.add_output("result", PortType.FLOAT)

    def execute(self, inputs):
        # Always guard None inputs
        if inputs["value"] is None:
            return {"result": None}
        try:
            return {"result": float(inputs["value"]) * int(inputs["factor"])}
        except Exception:
            return {"result": None}
```

### 2. Register it in nodes/__init__.py

```python
# At the top imports:
from nodes import my_nodes

# In _discover() calls (or add alongside existing ones):
_discover(my_nodes)
```

If it's a new category, add to CATEGORY_ORDER:
```python
CATEGORY_ORDER = [..., "MyCategory", ...]
```

### 3. Add category color in gui/theme.py

```python
MY_COLOR = (100, 200, 100)   # RGB tuple, no alpha
# Add to CATEGORY_COLORS dict:
"MyCategory": MY_COLOR,
```

### 4. Write a test

```python
# tests/test_my_nodes.py
def test_my_node():
    from nodes.my_nodes import MyNode
    r = MyNode().execute({"value": 3.0, "factor": 4})
    assert r["result"] == 12.0

def test_my_node_none_guard():
    from nodes.my_nodes import MyNode
    r = MyNode().execute({"value": None, "factor": 2})
    assert r["result"] is None
```

### 5. Run tests
```bash
cd C:/Users/lucas/node-tool-v3
python -m pytest tests/ -q
```

That's it. The palette, inspector, save/load, copy/paste, and undo/redo all work automatically.

---

## How to Add a New Port Type

1. **core/port.py** — add to enum: `MY_TYPE = auto()`
2. Add `default_value()` case: usually `return None`
3. Add `coerce()` case: usually `return value` (pass-through)
4. **gui/theme.py** — add `MY_TYPE_PIN = (R, G, B, 255)`
5. **gui/app.py** — add to `PIN_COLORS` and `PIN_SHAPES` dicts
6. If it's a reference type (not editable via widget), add to the reference-type check in `_create_input_widget` so it shows `"← connect"` instead of an editable widget

---

## Inline Visualization (IMAGE type)

Any node that outputs `PortType.IMAGE` (RGB uint8 ndarray, shape HxWx3) gets its output rendered as a live texture directly inside the node on the canvas after each Run.

To make a new viz node:
```python
from nodes.viz_nodes import _render_fig  # helper: fig → RGB ndarray

class MyPlotNode(BaseNode):
    type_name = "my_plot"
    label = "My Plot"
    category = "Viz"

    def _setup_ports(self):
        self.add_input("data", PortType.NDARRAY)
        self.add_output("image", PortType.IMAGE)

    def execute(self, inputs):
        if inputs["data"] is None:
            return {"image": None}
        import matplotlib.pyplot as plt
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        fig.patch.set_facecolor("#0A0A10")
        ax.set_facecolor("#0F0F1A")
        ax.plot(inputs["data"])
        return {"image": _render_fig(fig)}
```

---

## Training Panel Usage

1. Wire: `Dataset → TrainingConfig ← Sequential ← [layers]`
2. Wire: `TrainingConfig ← Optimizer ← [model]`
3. Wire: `TrainingConfig ← LossFunction`
4. Click **▶ Start** in Training Panel
5. Watch live loss curve update per epoch
6. Click **💾 Save Model** when done

The training runs in a background thread — UI stays fully interactive.

---

## Key Files to Know

| File | When to touch it |
|------|-----------------|
| `nodes/pytorch_nodes.py` | Add new ML layers, losses, optimizers |
| `nodes/viz_nodes.py` | Add new plot types |
| `nodes/numpy_nodes.py` | Add array operations |
| `nodes/pandas_nodes.py` | Add DataFrame operations |
| `nodes/sklearn_nodes.py` | Add ML models / metrics |
| `gui/training_panel.py` | Change training loop behavior |
| `gui/theme.py` | Change colors, add new category colors |
| `gui/app.py` | Change GUI layout, panels, keyboard shortcuts |
| `core/graph.py` | Change execution engine |

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Delete` | Delete selected nodes or links |
| `Ctrl+Z` | Undo |
| `Ctrl+Y` | Redo |
| `Ctrl+C` | Copy selected nodes |
| `Ctrl+V` | Paste nodes |
| `Ctrl+S` | Save graph to JSON |
| `Ctrl+O` | Load graph from JSON |
| Right-click | Context menu — add any node |

---

## Ideas for Next Sessions

- **Validation split** — second dataloader in TrainingConfig, val loss on plot
- **Model summary node** — show param count, layer shapes
- **Tensor image preview** — VizImageGridNode already exists for MNIST grids
- **Custom dataset node** — load from image folder or arbitrary CSV
- **Export to .py** — generate a clean runnable Python script from the graph
- **Subgraph / Group node** — collapse a selection into a reusable macro
- **LR scheduler nodes** — StepLR, CosineAnnealing wired into training
- **Evaluation node** — accuracy/F1 on test set after training
- **More Sklearn** — GridSearchCV, Pipeline, FeatureImportance
- **Live node execution** — re-run on every value change, not just on Run button
- **Node comments / sticky notes** — annotate sections of the canvas
- **Dark/light theme toggle**
