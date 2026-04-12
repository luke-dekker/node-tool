"""SO-101 Imitation Learning — ACT policy built from primitive layer nodes.

Marker-based architecture: no data-loading node in the graph. Three A markers
inject the state, image, and action tensors at training time. All dataset
config (path, batch_size, chunk_size, task_id) lives in the Training Panel.

The whole ACT architecture is visible in the graph. No black-box policy node.
Swap Conv2d for a ResNet, swap TransformerEncoderLayer for an LSTM, change
the fusion strategy — the graph IS the model, so all of it is yours to edit.

Architecture (training branch):

  Data In (A:observation.images.top) → Conv2d → ReLU → Conv2d → ReLU
          → AdaptiveAvgPool(4) → Flatten → Linear(→EMBED) ──┐
                                                           Unsqueeze(1)
                                                             │
  Data In (A:observation.state) → Linear(→EMBED) ─ Unsqueeze(1)┤
                                                             ↓
                                             Concat(dim=1) → (B, 2, EMBED)
                                             → PositionalEncoding
                                             → TransformerEncoderLayer
                                             → Flatten → Linear(→CHUNK*ACT)
                                             → Reshape(B, CHUNK, ACT)
                                                     │
  Data In (A:action) ──────────────────────────────── │
                                                      ↓
                                                LossCompute(mse)
                                                      ↓
                                                Data Out (B:loss)
"""
from __future__ import annotations
from core.graph import Graph
from templates._helpers import grid


LABEL = "SO-101 Imitation Learning"
DESCRIPTION = (
    "Train an ACT policy built entirely from primitive layer nodes. The "
    "transformer, vision encoder, state encoder, fusion, and action head "
    "are all visible in the graph — swap any of them for a different "
    "architecture. Marker-based — dataset lives in the panel."
)

# SO-101 specifics
_DOF    = 6    # 6 joints
_CHUNK  = 16   # predict 16 future actions at once
_EMBED  = 256  # per-modality embedding size
_GROUP  = "so101"


def build(graph: Graph) -> dict[str, tuple[int, int]]:
    from nodes.pytorch.input_marker                import InputMarkerNode
    from nodes.pytorch.conv2d                      import Conv2dNode
    from nodes.pytorch.adaptive_avgpool2d          import AdaptiveAvgPool2dNode
    from nodes.pytorch.flatten                     import FlattenNode
    from nodes.pytorch.linear                      import LinearNode
    from nodes.pytorch.tensor_shape_op             import TensorShapeOpNode
    from nodes.pytorch.tensor_cat                  import TensorCatNode
    from nodes.pytorch.positional_encoding         import PositionalEncodingNode
    from nodes.pytorch.transformer_encoder_layer   import TransformerEncoderLayerNode
    # TensorShapeOpNode already imported above
    from nodes.pytorch.loss_compute                import LossComputeNode
    from nodes.pytorch.train_marker                import TrainMarkerNode

    pos = grid(step_x=240, step_y=170)
    positions: dict[str, tuple[int, int]] = {}

    # ── Input markers ────────────────────────────────────────────────────────
    image_in = InputMarkerNode()
    image_in.inputs["modality"].default_value = "observation.images.top"
    image_in.inputs["group"].default_value    = _GROUP
    graph.add_node(image_in)
    positions[image_in.id] = pos(0, 0)

    state_in = InputMarkerNode()
    state_in.inputs["modality"].default_value = "observation.state"
    state_in.inputs["group"].default_value    = _GROUP
    graph.add_node(state_in)
    positions[state_in.id] = pos(0, 2)

    action_in = InputMarkerNode()
    action_in.inputs["modality"].default_value = "action"
    action_in.inputs["group"].default_value    = _GROUP
    graph.add_node(action_in)
    positions[action_in.id] = pos(0, 3)

    # ── Vision encoder branch ────────────────────────────────────────────────
    conv1 = Conv2dNode()
    conv1.inputs["in_ch"].default_value      = 3
    conv1.inputs["out_ch"].default_value     = 32
    conv1.inputs["kernel"].default_value     = 5
    conv1.inputs["stride"].default_value     = 2
    conv1.inputs["padding"].default_value    = 2
    conv1.inputs["activation"].default_value = "relu"
    graph.add_node(conv1)
    positions[conv1.id] = pos(1, 0)

    conv2 = Conv2dNode()
    conv2.inputs["in_ch"].default_value      = 32
    conv2.inputs["out_ch"].default_value     = 64
    conv2.inputs["kernel"].default_value     = 5
    conv2.inputs["stride"].default_value     = 2
    conv2.inputs["padding"].default_value    = 2
    conv2.inputs["activation"].default_value = "relu"
    graph.add_node(conv2)
    positions[conv2.id] = pos(2, 0)

    pool = AdaptiveAvgPool2dNode()
    pool.inputs["output_size"].default_value = 4  # (B, 64, 4, 4) = 1024 features
    graph.add_node(pool)
    positions[pool.id] = pos(3, 0)

    vflat = FlattenNode()
    graph.add_node(vflat)
    positions[vflat.id] = pos(4, 0)

    vlin = LinearNode()
    vlin.inputs["in_features"].default_value  = 64 * 4 * 4  # 1024
    vlin.inputs["out_features"].default_value = _EMBED
    vlin.inputs["activation"].default_value   = "relu"
    graph.add_node(vlin)
    positions[vlin.id] = pos(5, 0)

    vunsq = TensorShapeOpNode()
    vunsq.inputs["mode"].default_value = "unsqueeze"
    vunsq.inputs["dim"].default_value = 1  # (B, EMBED) → (B, 1, EMBED)
    graph.add_node(vunsq)
    positions[vunsq.id] = pos(6, 0)

    # Wire vision
    graph.add_connection(image_in.id, "tensor",     conv1.id, "tensor_in")
    graph.add_connection(conv1.id,    "tensor_out", conv2.id, "tensor_in")
    graph.add_connection(conv2.id,    "tensor_out", pool.id,  "tensor_in")
    graph.add_connection(pool.id,     "tensor_out", vflat.id, "tensor_in")
    graph.add_connection(vflat.id,    "tensor_out", vlin.id,  "tensor_in")
    graph.add_connection(vlin.id,     "tensor_out", vunsq.id, "tensor")

    # ── State encoder branch ─────────────────────────────────────────────────
    slin = LinearNode()
    slin.inputs["in_features"].default_value  = _DOF
    slin.inputs["out_features"].default_value = _EMBED
    slin.inputs["activation"].default_value   = "relu"
    graph.add_node(slin)
    positions[slin.id] = pos(5, 2)

    sunsq = TensorShapeOpNode()
    sunsq.inputs["mode"].default_value = "unsqueeze"
    sunsq.inputs["dim"].default_value = 1  # (B, EMBED) → (B, 1, EMBED)
    graph.add_node(sunsq)
    positions[sunsq.id] = pos(6, 2)

    graph.add_connection(state_in.id, "tensor",     slin.id,  "tensor_in")
    graph.add_connection(slin.id,     "tensor_out", sunsq.id, "tensor")

    # ── Fusion (concat along sequence dim) ───────────────────────────────────
    fuse = TensorCatNode()
    fuse.inputs["dim"].default_value = 1  # concat along T → (B, 2, EMBED)
    graph.add_node(fuse)
    positions[fuse.id] = pos(7, 1)

    graph.add_connection(vunsq.id, "tensor", fuse.id, "t1")
    graph.add_connection(sunsq.id, "tensor", fuse.id, "t2")

    # ── Transformer encoder (1 block; stack more by duplicating) ─────────────
    pe = PositionalEncodingNode()
    pe.inputs["d_model"].default_value = _EMBED
    pe.inputs["max_len"].default_value = 8
    pe.inputs["kind"].default_value    = "learned"
    graph.add_node(pe)
    positions[pe.id] = pos(8, 1)

    tel = TransformerEncoderLayerNode()
    tel.inputs["d_model"].default_value         = _EMBED
    tel.inputs["nhead"].default_value           = 8
    tel.inputs["dim_feedforward"].default_value = 512
    tel.inputs["dropout"].default_value         = 0.1
    graph.add_node(tel)
    positions[tel.id] = pos(9, 1)

    graph.add_connection(fuse.id, "tensor",     pe.id,  "tensor_in")
    graph.add_connection(pe.id,   "tensor_out", tel.id, "tensor_in")

    # ── Action head ──────────────────────────────────────────────────────────
    head_flat = FlattenNode()  # (B, 2, EMBED) → (B, 2*EMBED)
    graph.add_node(head_flat)
    positions[head_flat.id] = pos(10, 1)

    head_lin = LinearNode()
    head_lin.inputs["in_features"].default_value  = 2 * _EMBED
    head_lin.inputs["out_features"].default_value = _CHUNK * _DOF
    head_lin.inputs["activation"].default_value   = "none"
    graph.add_node(head_lin)
    positions[head_lin.id] = pos(11, 1)

    head_reshape = TensorShapeOpNode()
    head_reshape.inputs["mode"].default_value = "reshape"
    head_reshape.inputs["shape"].default_value = f"-1,{_CHUNK},{_DOF}"
    graph.add_node(head_reshape)
    positions[head_reshape.id] = pos(12, 1)

    graph.add_connection(tel.id,       "tensor_out", head_flat.id,    "tensor_in")
    graph.add_connection(head_flat.id, "tensor_out", head_lin.id,     "tensor_in")
    graph.add_connection(head_lin.id,  "tensor_out", head_reshape.id, "tensor")

    # ── Loss + train marker ───────────────────────────────────────────────────
    loss = LossComputeNode()
    loss.inputs["loss_type"].default_value = "mse"
    graph.add_node(loss)
    positions[loss.id] = pos(13, 1)

    graph.add_connection(head_reshape.id, "tensor", loss.id, "pred")
    graph.add_connection(action_in.id,    "tensor", loss.id, "target")

    data_out = TrainMarkerNode()
    data_out.inputs["kind"].default_value      = "loss"
    data_out.inputs["group"].default_value     = _GROUP
    data_out.inputs["task_name"].default_value = "so101_train"
    graph.add_node(data_out)
    positions[data_out.id] = pos(14, 1)

    graph.add_connection(loss.id, "loss", data_out.id, "tensor_in")

    return positions
