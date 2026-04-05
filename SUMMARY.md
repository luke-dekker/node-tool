# node-tool-v5 — Project Summary

Successor to node-tool-v4. DearPyGui-based visual node editor for building, training, and fine-tuning PyTorch models. 348 tests passing.

**Core philosophy:** the graph IS the model. Each layer node owns persistent weights. Data flows left to right through real nn.Modules. Training updates those weights in-place.

## How to run
```
cd C:\Users\lucas\node-tool-v5
python main.py
```

## What changed from v4

### Core consolidation (9 files -> 4)
| Before | After |
|---|---|
| `graph.py` + `commands.py` | `graph.py` |
| `node.py` + `port.py` | `node.py` |
| `serializer.py` + `exporter.py` | `io.py` |
| `hot_reload.py` + `simple_node.py` | `custom.py` |
| `executor.py` | deleted |

Import paths updated everywhere:
- `from core.port import PortType` -> `from core.node import PortType`
- `from core.commands import ...` -> `from core.graph import ...`
- `from core.serializer/exporter import ...` -> `from core.io import ...`
- `from core.hot_reload/simple_node import ...` -> `from core.custom import ...`

### Inference panel removed
The GUI inference panel (manual shape/value input) is gone. Inference is done in the graph:
```
MNIST -> SampleBatch -> Load Model -> Argmax -> Print Tensor
```

### LoadModelNode (nodes/pytorch/persistence.py)
Fits the tensor-flow path AND participates in training. Key ports:
- `tensor_in` / `tensor_out`
- `path` — .pt file (reloads only when path changes)
- `freeze` — freeze all parameters
- `trainable_layers` — unfreeze last N child modules
- `save_path` — save model on every graph run if set
- `save_mode` — `overwrite` (replace same file) or `increment` (model_1.pt, model_2.pt ...)

`get_layers()` returns `[self._model]` so the training loop assembles it into the Sequential automatically.

## Architecture

### Core (4 files)
- `core/graph.py` — Graph, Connection, topological executor, CommandStack (undo/redo)
- `core/node.py` — BaseNode ABC, PortType enum, Port dataclass
- `core/io.py` — Serializer (JSON) + GraphExporter (Python script)
- `core/custom.py` — HotReloader (nodes/custom/ watcher) + @node decorator

### GUI (3 files)
- `gui/app.py` — NodeApp (~2000 lines)
- `gui/training_panel.py` — TrainingController (background thread, queue-based)
- `gui/theme.py` — colors, pin colors per PortType

### Nodes
- `nodes/pytorch/layers.py` — core layer nodes (LinearNode, Conv2dNode, etc.)
- `nodes/pytorch/training.py` — TrainingConfigNode, ForwardPassNode
- `nodes/pytorch/data.py` — SampleBatchNode, MNISTDatasetNode, ArgmaxNode, PrintTensorNode, etc.
- `nodes/pytorch/persistence.py` — LoadModelNode, SaveFullModelNode, PretrainedBlockNode, etc.
- `nodes/pytorch/` — also: optimizers, losses, schedulers, recurrent, autoencoder, backbones, architecture, tensor_ops, dataset_ops, dataset_sources, dataset_transforms, viz, pipeline

## The key design: persistent layer nodes

Each layer node holds `self._layer` (nn.Module). `get_layers()` returns `[self._layer]`.

`_collect_model_layers()` in app.py walks `tensor_in` connections backwards from TrainingConfig, calls `get_layers()` on each node, builds `nn.Sequential`.

After each epoch, `refresh_graph_silent()` re-executes the graph — `tensor_out` on every node reflects current weights.

Structural changes (e.g. `out_features`) recreate `self._layer` (reset weights). `freeze` never resets weights.

## Training flow

1. Build: `MNIST -> SampleBatch -> Flatten -> Linear(relu) -> Linear(relu) -> Linear -> TrainingConfig`
2. Run graph — tensors flow, tensor_out updates
3. Start Training -> assembles Sequential, builds optimizer, starts background thread
4. After each epoch: silent graph refresh shows live weights
5. Save Model -> tkinter Save As dialog -> .pt file

## Demo graph (pre-populated on launch)
```
MNIST(train) --dataloader--> TrainingConfig
             --dataloader--> SampleBatch --x--> Flatten --> Lin1(relu) --> Lin2(relu) --> Lin3 --> TrainingConfig(tensor_in)
MNIST(val)   --dataloader--> TrainingConfig
```
784 input, 256->128->10. CrossEntropy, Adam.

## Common gotchas

- **ArgmaxNode port is `tensor` not `tensor_in`**
- **PrintTensorNode** shows values when <=128 elements
- **LoadModelNode** caches model, reloads only when `path` changes
- **save_path on LoadModelNode** fires on EVERY graph run — leave blank during training

## Tests
```
python -m pytest tests/ -q   # 348 passed, 7 skipped
```
