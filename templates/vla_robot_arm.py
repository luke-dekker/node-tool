"""Vision-Language-Action (VLA) model with 6-DOF robot arm controller.

A simple VLA that fuses camera image, language command, and proprioceptive
joint state into a unified representation, then predicts 6 joint angle
deltas for a robot arm. The whole pipeline is visible in the graph:

Training branch (marker-based):

  Data In (A:observation.images.top)
      → Conv2d(3→32) → Conv2d(32→64) → AdaptiveAvgPool(4)
      → Flatten → Linear(1024→256) → Unsqueeze(1) ──────────┐
                                                              │
  Data In (A:observation.language)                            │
      → Embedding(1000, 64) → Linear(64→256) → Unsqueeze(1) ─┤
                                                              │ Cat(dim=1)
  Data In (A:observation.state)                               │  → (B,3,256)
      → Linear(6→256) → Unsqueeze(1) ────────────────────────┘
                                                   ↓
                                        PositionalEncoding
                                        TransformerEncoderLayer
                                        Flatten → Linear(→96)
                                        Reshape(B,CHUNK,6)
                                              │
  Data In (A:action) ─────────────────────────│
                                              ↓
                                        LossCompute(mse)
                                              ↓
                                        Data Out (B:loss)

Deployment branch (robot arm output):

  Action head output → [split to 6 channels via reshape]
      → SafetyLimiter per joint
      → Servo Bus (feetech/dynamixel/PWM)

After training, disconnect the loss/marker nodes and wire the action head
output into the servo bus for real-time inference on hardware.
"""
from __future__ import annotations
from core.graph import Graph
from templates._helpers import grid


LABEL = "VLA + 6-DOF Robot Arm"
DESCRIPTION = (
    "Vision-Language-Action model that fuses camera, language command, and "
    "joint state via a transformer, then outputs 6-DOF joint actions. "
    "Includes a servo bus node for hardware deployment."
)

_DOF    = 6
_CHUNK  = 1      # predict 1 step (no chunking — simple VLA)
_EMBED  = 256
_VOCAB  = 1000   # vocabulary size for language embedding
_LANG_D = 64     # embedding dimension
_GROUP  = "vla_arm"

# Joint names for a standard 6-DOF arm
_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]


def build(graph: Graph) -> dict[str, tuple[int, int]]:
    from nodes.pytorch.input_marker                import InputMarkerNode
    from nodes.pytorch.conv2d                      import Conv2dNode
    from nodes.pytorch.adaptive_avgpool2d          import AdaptiveAvgPool2dNode
    from nodes.pytorch.flatten                     import FlattenNode
    from nodes.pytorch.linear                      import LinearNode
    from nodes.pytorch.embedding                   import EmbeddingNode
    from nodes.pytorch.tensor_shape_op             import TensorShapeOpNode
    from nodes.pytorch.tensor_cat                  import TensorCatNode
    from nodes.pytorch.positional_encoding         import PositionalEncodingNode
    from nodes.pytorch.transformer_encoder_layer   import TransformerEncoderLayerNode
    from nodes.pytorch.loss_compute                import LossComputeNode
    from nodes.pytorch.train_marker                import TrainMarkerNode
    from plugins.robotics.nodes.trajectory         import SafetyLimiterNode
    from plugins.robotics.nodes.motors             import ServoBusNode

    pos = grid(step_x=240, step_y=160)
    positions: dict[str, tuple[int, int]] = {}

    # ── Input markers ────────────────────────────────────────────────────
    image_in = InputMarkerNode()
    image_in.inputs["modality"].default_value = "observation.images.top"
    image_in.inputs["group"].default_value    = _GROUP
    graph.add_node(image_in)
    positions[image_in.id] = pos(0, 0)

    lang_in = InputMarkerNode()
    lang_in.inputs["modality"].default_value = "observation.language"
    lang_in.inputs["group"].default_value    = _GROUP
    graph.add_node(lang_in)
    positions[lang_in.id] = pos(0, 2)

    state_in = InputMarkerNode()
    state_in.inputs["modality"].default_value = "observation.state"
    state_in.inputs["group"].default_value    = _GROUP
    graph.add_node(state_in)
    positions[state_in.id] = pos(0, 4)

    action_in = InputMarkerNode()
    action_in.inputs["modality"].default_value = "action"
    action_in.inputs["group"].default_value    = _GROUP
    graph.add_node(action_in)
    positions[action_in.id] = pos(0, 6)

    # ── Vision encoder branch ────────────────────────────────────────────
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
    conv2.inputs["kernel"].default_value     = 3
    conv2.inputs["stride"].default_value     = 2
    conv2.inputs["padding"].default_value    = 1
    conv2.inputs["activation"].default_value = "relu"
    graph.add_node(conv2)
    positions[conv2.id] = pos(2, 0)

    vpool = AdaptiveAvgPool2dNode()
    vpool.inputs["output_size"].default_value = 4  # (B, 64, 4, 4) = 1024
    graph.add_node(vpool)
    positions[vpool.id] = pos(3, 0)

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
    vunsq.inputs["dim"].default_value  = 1  # (B, EMBED) → (B, 1, EMBED)
    graph.add_node(vunsq)
    positions[vunsq.id] = pos(6, 0)

    # Wire vision
    graph.add_connection(image_in.id, "tensor",     conv1.id,  "tensor_in")
    graph.add_connection(conv1.id,    "tensor_out", conv2.id,  "tensor_in")
    graph.add_connection(conv2.id,    "tensor_out", vpool.id,  "tensor_in")
    graph.add_connection(vpool.id,    "tensor_out", vflat.id,  "tensor_in")
    graph.add_connection(vflat.id,    "tensor_out", vlin.id,   "tensor_in")
    graph.add_connection(vlin.id,     "tensor_out", vunsq.id,  "tensor")

    # ── Language encoder branch ──────────────────────────────────────────
    embed = EmbeddingNode()
    embed.inputs["num_embeddings"].default_value = _VOCAB
    embed.inputs["embedding_dim"].default_value  = _LANG_D
    graph.add_node(embed)
    positions[embed.id] = pos(2, 2)

    # Flatten the embedded tokens: (B, seq, 64) → (B, seq*64)
    lflat = FlattenNode()
    graph.add_node(lflat)
    positions[lflat.id] = pos(3, 2)

    llin = LinearNode()
    llin.inputs["in_features"].default_value  = _LANG_D  # adjusted at runtime
    llin.inputs["out_features"].default_value = _EMBED
    llin.inputs["activation"].default_value   = "relu"
    graph.add_node(llin)
    positions[llin.id] = pos(5, 2)

    lunsq = TensorShapeOpNode()
    lunsq.inputs["mode"].default_value = "unsqueeze"
    lunsq.inputs["dim"].default_value  = 1
    graph.add_node(lunsq)
    positions[lunsq.id] = pos(6, 2)

    # Wire language
    graph.add_connection(lang_in.id, "tensor",     embed.id,  "tensor_in")
    graph.add_connection(embed.id,   "tensor_out", lflat.id,  "tensor_in")
    graph.add_connection(lflat.id,   "tensor_out", llin.id,   "tensor_in")
    graph.add_connection(llin.id,    "tensor_out", lunsq.id,  "tensor")

    # ── State encoder branch ────────────────────────────────────────────
    slin = LinearNode()
    slin.inputs["in_features"].default_value  = _DOF
    slin.inputs["out_features"].default_value = _EMBED
    slin.inputs["activation"].default_value   = "relu"
    graph.add_node(slin)
    positions[slin.id] = pos(5, 4)

    sunsq = TensorShapeOpNode()
    sunsq.inputs["mode"].default_value = "unsqueeze"
    sunsq.inputs["dim"].default_value  = 1
    graph.add_node(sunsq)
    positions[sunsq.id] = pos(6, 4)

    graph.add_connection(state_in.id, "tensor",     slin.id,  "tensor_in")
    graph.add_connection(slin.id,     "tensor_out", sunsq.id, "tensor")

    # ── Fusion (concat along sequence dim) ───────────────────────────────
    fuse = TensorCatNode()
    fuse.inputs["dim"].default_value = 1  # (B, 3, EMBED) — vision + lang + state
    graph.add_node(fuse)
    positions[fuse.id] = pos(7, 2)

    graph.add_connection(vunsq.id, "tensor", fuse.id, "t1")
    graph.add_connection(lunsq.id, "tensor", fuse.id, "t2")
    graph.add_connection(sunsq.id, "tensor", fuse.id, "t3")

    # ── Transformer encoder ──────────────────────────────────────────────
    pe = PositionalEncodingNode()
    pe.inputs["d_model"].default_value = _EMBED
    pe.inputs["max_len"].default_value = 8
    pe.inputs["kind"].default_value    = "learned"
    graph.add_node(pe)
    positions[pe.id] = pos(8, 2)

    tel = TransformerEncoderLayerNode()
    tel.inputs["d_model"].default_value         = _EMBED
    tel.inputs["nhead"].default_value           = 8
    tel.inputs["dim_feedforward"].default_value = 512
    tel.inputs["dropout"].default_value         = 0.1
    graph.add_node(tel)
    positions[tel.id] = pos(9, 2)

    graph.add_connection(fuse.id, "tensor",     pe.id,  "tensor_in")
    graph.add_connection(pe.id,   "tensor_out", tel.id, "tensor_in")

    # ── Action head ──────────────────────────────────────────────────────
    head_flat = FlattenNode()  # (B, 3, EMBED) → (B, 3*EMBED)
    graph.add_node(head_flat)
    positions[head_flat.id] = pos(10, 2)

    head_lin = LinearNode()
    head_lin.inputs["in_features"].default_value  = 3 * _EMBED
    head_lin.inputs["out_features"].default_value = _CHUNK * _DOF
    head_lin.inputs["activation"].default_value   = "none"
    graph.add_node(head_lin)
    positions[head_lin.id] = pos(11, 2)

    # Reshape to (B, CHUNK, 6) — one action per chunk step, 6 joints
    head_reshape = TensorShapeOpNode()
    head_reshape.inputs["mode"].default_value  = "reshape"
    head_reshape.inputs["shape"].default_value = f"-1,{_CHUNK},{_DOF}"
    graph.add_node(head_reshape)
    positions[head_reshape.id] = pos(12, 2)

    graph.add_connection(tel.id,          "tensor_out", head_flat.id,    "tensor_in")
    graph.add_connection(head_flat.id,    "tensor_out", head_lin.id,     "tensor_in")
    graph.add_connection(head_lin.id,     "tensor_out", head_reshape.id, "tensor")

    # ── Loss + train marker ──────────────────────────────────────────────
    loss = LossComputeNode()
    loss.inputs["loss_type"].default_value = "mse"
    graph.add_node(loss)
    positions[loss.id] = pos(13, 3)

    graph.add_connection(head_reshape.id, "tensor", loss.id, "pred")
    graph.add_connection(action_in.id,    "tensor", loss.id, "target")

    data_out = TrainMarkerNode()
    data_out.inputs["kind"].default_value      = "loss"
    data_out.inputs["group"].default_value     = _GROUP
    data_out.inputs["task_name"].default_value = "vla_train"
    graph.add_node(data_out)
    positions[data_out.id] = pos(14, 3)

    graph.add_connection(loss.id, "loss", data_out.id, "tensor_in")

    # ── Robot arm controller (deployment branch) ─────────────────────────
    # Squeeze the chunk dim: (B, 1, 6) → (B, 6) for single-step inference
    squeeze = TensorShapeOpNode()
    squeeze.inputs["mode"].default_value  = "reshape"
    squeeze.inputs["shape"].default_value = f"-1,{_DOF}"
    graph.add_node(squeeze)
    positions[squeeze.id] = pos(13, 5)

    graph.add_connection(head_reshape.id, "tensor", squeeze.id, "tensor")

    # Servo Bus — connects to real hardware for deployment.
    # target_positions accepts a tensor/list of joint values.
    # Set write=True and configure the serial port to send to hardware.
    servo_bus = ServoBusNode()
    servo_bus.inputs["protocol"].default_value    = "feetech_sts"
    servo_bus.inputs["port"].default_value        = ""
    servo_bus.inputs["baud_rate"].default_value   = 1000000
    servo_bus.inputs["joint_names"].default_value = ",".join(_JOINTS)
    servo_bus.inputs["joint_ids"].default_value   = "1,2,3,4,5,6"
    servo_bus.inputs["model"].default_value       = "sts3215"
    servo_bus.inputs["write"].default_value       = False
    graph.add_node(servo_bus)
    positions[servo_bus.id] = pos(14, 5)

    graph.add_connection(squeeze.id, "tensor", servo_bus.id, "target_positions")

    return positions
