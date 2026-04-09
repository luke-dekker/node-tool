"""Multi-modal classification template — fusion of audio + image with loss-as-output.

Demonstrates:

  - Multi-modal training with multiple input modalities (audio + image)
  - Per-modality encoder chains, all visible as graph nodes
  - Concat fusion via tensor_cat
  - Single classification head over the fused features
  - Loss computed inside the graph via LossComputeNode + cross_entropy
  - TrainingConfig in loss_is_output mode — no special multimodal loop needed

Architecture:

    [Inline-generated multimodal demo data, both modalities populated]
                            |
                  FolderMultimodalDataset (audio + image)
                            |
                      MultiDataset (provides multimodal_collate)
                            |
                       BatchInput
                       /        \\
                  audio          image
                    |              |
              Linear+ReLU       Flatten
                    |              |
              Linear+ReLU       Linear+ReLU
                    \\            /
                     tensor_cat (concat 16 + 16 → 32)
                            |
                  Linear (classification head, 32 → 2)
                            |
                  LossCompute(cross_entropy, head, label)
                            |
                  TrainingConfig(loss_is_output=True)

Both encoders + the head train via standard backprop. The same pattern
extends to multi-task: add more LossComputeNodes for different heads,
sum them via tensor_add, feed the combined scalar to TrainingConfig.

The build() function generates a tiny synthetic 2-class dataset on disk
the first time it runs, so the template trains out of the box without
requiring the user to provide their own multimodal data.
"""
from __future__ import annotations
from core.graph import Graph
from templates._helpers import grid

LABEL = "Multi-Modal Classification"
DESCRIPTION = (
    "Fusion of audio + image with per-modality encoders, concat fusion, and "
    "loss-as-output. The same wiring pattern scales to multi-task. Auto-"
    "generates a small demo dataset on first run."
)

DEMO_DIR = "demo_data/multimodal_full"
N_SAMPLES = 40
N_CLASSES = 2


def _ensure_multimodal_full_dataset() -> None:
    """Generate a tiny synthetic dataset where EVERY sample has BOTH modalities.

    Differs from the existing _ensure_demo_multimodal_data in edit_ops.py: this
    one populates audio AND image for each sample, so multimodal_collate keeps
    both modalities in every batch instead of dropping one as ragged.

    Audio is a (64,) float vector with class-dependent mean.
    Image is an (8, 8, 3) uint8 RGB tile with class-dependent color.
    Both are stored alongside a samples.csv manifest pointing at them.
    """
    import os
    if os.path.isdir(DEMO_DIR):
        return  # already generated
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return  # pillow / numpy missing — caller will see a load error at runtime

    os.makedirs(os.path.join(DEMO_DIR, "audio"), exist_ok=True)
    os.makedirs(os.path.join(DEMO_DIR, "image"), exist_ok=True)
    rows = ["id,label,audio,image"]
    for i in range(N_SAMPLES):
        label = i % N_CLASSES
        # Audio: class-conditional mean shift
        audio = np.random.randn(64).astype(np.float32) + (label * 2.0)
        np.save(os.path.join(DEMO_DIR, "audio", f"{i:03d}.npy"), audio)
        # Image: class-conditional dominant color
        base = np.zeros((8, 8, 3), dtype=np.uint8)
        base[:] = (label * 200, 50, 100)
        base += np.random.randint(0, 30, base.shape, dtype=np.uint8)
        Image.fromarray(base).save(os.path.join(DEMO_DIR, "image", f"{i:03d}.png"))
        rows.append(f"{i:03d},{label},{i:03d}.npy,{i:03d}.png")
    with open(os.path.join(DEMO_DIR, "samples.csv"), "w") as f:
        f.write("\n".join(rows))


def build(graph: Graph) -> dict[str, tuple[int, int]]:
    _ensure_multimodal_full_dataset()

    from nodes.pytorch.folder_multimodal_dataset import FolderMultimodalDatasetNode
    from nodes.pytorch.multi_dataset              import MultiDatasetNode
    from nodes.pytorch.batch_input                import BatchInputNode
    from nodes.pytorch.linear                     import LinearNode
    from nodes.pytorch.flatten                    import FlattenNode
    from nodes.pytorch.tensor_cat                 import TensorCatNode
    from nodes.pytorch.loss_compute               import LossComputeNode
    from nodes.pytorch.training_config            import TrainingConfigNode

    pos = grid(step_x=220)
    positions: dict[str, tuple[int, int]] = {}

    # ── Data ────────────────────────────────────────────────────────────────
    ds = FolderMultimodalDatasetNode()
    ds.inputs["root_path"].default_value  = DEMO_DIR
    ds.inputs["modalities"].default_value = "audio,image"
    graph.add_node(ds); positions[ds.id] = pos(col=0, row=2)

    multi = MultiDatasetNode()
    multi.inputs["batch_size"].default_value = 8
    multi.inputs["shuffle"].default_value    = True
    graph.add_node(multi); positions[multi.id] = pos(col=1, row=2)

    batch = BatchInputNode()
    graph.add_node(batch); positions[batch.id] = pos(col=2, row=2)

    # ── Audio encoder: 64 → 32 → 16 ─────────────────────────────────────────
    audio_lin1 = LinearNode()
    audio_lin1.inputs["in_features"].default_value  = 64
    audio_lin1.inputs["out_features"].default_value = 32
    audio_lin1.inputs["activation"].default_value   = "relu"
    graph.add_node(audio_lin1); positions[audio_lin1.id] = pos(col=3, row=1)

    audio_lin2 = LinearNode()
    audio_lin2.inputs["in_features"].default_value  = 32
    audio_lin2.inputs["out_features"].default_value = 16
    audio_lin2.inputs["activation"].default_value   = "relu"
    graph.add_node(audio_lin2); positions[audio_lin2.id] = pos(col=4, row=1)

    # ── Image encoder: 8x8x3 → flatten → 192 → 32 → 16 ──────────────────────
    img_flat = FlattenNode()
    img_flat.inputs["start_dim"].default_value = 1
    graph.add_node(img_flat); positions[img_flat.id] = pos(col=3, row=3)

    img_lin1 = LinearNode()
    img_lin1.inputs["in_features"].default_value  = 192
    img_lin1.inputs["out_features"].default_value = 32
    img_lin1.inputs["activation"].default_value   = "relu"
    graph.add_node(img_lin1); positions[img_lin1.id] = pos(col=4, row=3)

    img_lin2 = LinearNode()
    img_lin2.inputs["in_features"].default_value  = 32
    img_lin2.inputs["out_features"].default_value = 16
    img_lin2.inputs["activation"].default_value   = "relu"
    graph.add_node(img_lin2); positions[img_lin2.id] = pos(col=5, row=3)

    # ── Fusion: concat audio[16] + image[16] → 32 ───────────────────────────
    fuse = TensorCatNode()
    fuse.inputs["dim"].default_value = 1   # batch dim is 0, feature dim is 1
    graph.add_node(fuse); positions[fuse.id] = pos(col=6, row=2)

    # ── Classification head: 32 → 2 ─────────────────────────────────────────
    head = LinearNode()
    head.inputs["in_features"].default_value  = 32
    head.inputs["out_features"].default_value = N_CLASSES
    head.inputs["activation"].default_value   = "none"
    graph.add_node(head); positions[head.id] = pos(col=7, row=2)

    # ── Loss inside the graph ───────────────────────────────────────────────
    loss = LossComputeNode()
    loss.inputs["loss_type"].default_value = "cross_entropy"
    graph.add_node(loss); positions[loss.id] = pos(col=8, row=2)

    # ── Training config in loss-as-output mode ──────────────────────────────
    cfg = TrainingConfigNode()
    cfg.inputs["epochs"].default_value         = 10
    cfg.inputs["lr"].default_value             = 0.005
    cfg.inputs["optimizer"].default_value      = "adam"
    cfg.inputs["loss_is_output"].default_value = True
    graph.add_node(cfg); positions[cfg.id] = pos(col=9, row=2)

    # ── Wire it ─────────────────────────────────────────────────────────────
    # Data flow
    graph.add_connection(ds.id,    "dataset",    multi.id, "dataset_1")
    graph.add_connection(multi.id, "dataloader", batch.id, "dataloader")
    graph.add_connection(multi.id, "dataloader", cfg.id,   "dataloader")

    # Audio chain
    graph.add_connection(batch.id,     "audio",      audio_lin1.id, "tensor_in")
    graph.add_connection(audio_lin1.id,"tensor_out", audio_lin2.id, "tensor_in")

    # Image chain
    graph.add_connection(batch.id,    "image",      img_flat.id, "tensor_in")
    graph.add_connection(img_flat.id, "tensor_out", img_lin1.id, "tensor_in")
    graph.add_connection(img_lin1.id, "tensor_out", img_lin2.id, "tensor_in")

    # Fusion: TensorCatNode takes t1..t4 inputs; wire audio_lin2 → t1 and img_lin2 → t2
    graph.add_connection(audio_lin2.id, "tensor_out", fuse.id, "t1")
    graph.add_connection(img_lin2.id,   "tensor_out", fuse.id, "t2")

    # Head + loss
    graph.add_connection(fuse.id, "tensor",     head.id, "tensor_in")
    graph.add_connection(head.id, "tensor_out", loss.id, "pred")
    graph.add_connection(batch.id, "label",     loss.id, "target")

    # Loss → training config
    graph.add_connection(loss.id, "loss",       cfg.id,  "tensor_in")
    return positions
