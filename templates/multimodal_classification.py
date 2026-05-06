"""Multi-modal classification — audio + image fusion with loss-as-output.

Marker-based architecture: the graph contains zero data-loading nodes. Three A
markers inject audio, image, and label batches at training time; per-modality
encoders fuse their outputs; cross-entropy loss is computed in the graph; and a
B marker marks the loss as the training output. All dataset config lives in the
Training Panel.

    Data In (A:audio) → Linear → ─┐
    Data In (A:image) → Flatten → Linear → ─┤ TensorCat → Linear → LossCompute → Data Out (B:loss)
    Data In (A:label) ──────────────────────────────────────────────────────────↗ (target)
"""
from __future__ import annotations
from core.graph import Graph
from templates._helpers import grid

LABEL = "Multi-Modal Classification"
DESCRIPTION = "Audio + image fusion with per-modality encoders, concat, loss-as-output. Marker-based — dataset lives in the panel."

DEMO_DIR = "demo_data/multimodal_full"

def _ensure_data():
    import os
    if os.path.isdir(DEMO_DIR): return
    try:
        import numpy as np; from PIL import Image
    except ImportError: return
    os.makedirs(f"{DEMO_DIR}/audio", exist_ok=True); os.makedirs(f"{DEMO_DIR}/image", exist_ok=True)
    rows = ["id,audio,image,label"]
    for i in range(40):
        label = "cat" if i%2==0 else "dog"
        np.save(f"{DEMO_DIR}/audio/{i:03d}.npy", np.random.randn(64).astype(np.float32) + (i%2)*2)
        img = np.zeros((8,8,3),dtype=np.uint8); img[:] = ((i%2)*200,50,100)
        img += np.random.randint(0,30,img.shape,dtype=np.uint8)
        Image.fromarray(img).save(f"{DEMO_DIR}/image/{i:03d}.png")
        rows.append(f"{i:03d},audio/{i:03d}.npy,image/{i:03d}.png,{label}")
    with open(f"{DEMO_DIR}/samples.csv","w") as f: f.write("\n".join(rows))

def build(graph: Graph) -> dict[str, tuple[int, int]]:
    from nodes.pytorch.input_marker  import InputMarkerNode
    from nodes.pytorch.layer         import LayerNode
    from nodes.pytorch.flatten       import FlattenNode
    from nodes.pytorch.tensor_reshape import TensorReshapeNode
    from nodes.pytorch.loss_compute  import LossComputeNode
    from nodes.pytorch.train_marker  import TrainMarkerNode

    def _lin(out_f, act="none"):
        n = LayerNode()
        n.inputs["kind"].default_value         = "linear"
        n.inputs["out_features"].default_value = out_f
        n.inputs["activation"].default_value   = act
        return n

    pos = grid(step_x=220); positions = {}

    audio_in = InputMarkerNode(); audio_in.inputs["modality"].default_value = "audio"
    graph.add_node(audio_in); positions[audio_in.id] = pos(col=0, row=1)

    image_in = InputMarkerNode(); image_in.inputs["modality"].default_value = "image"
    graph.add_node(image_in); positions[image_in.id] = pos(col=0, row=3)

    label_in = InputMarkerNode(); label_in.inputs["modality"].default_value = "label"
    graph.add_node(label_in); positions[label_in.id] = pos(col=0, row=5)

    a1 = _lin(16, "relu"); graph.add_node(a1); positions[a1.id] = pos(col=1, row=1)

    img_flat = FlattenNode(); graph.add_node(img_flat); positions[img_flat.id] = pos(col=1, row=3)
    i1 = _lin(16, "relu"); graph.add_node(i1); positions[i1.id] = pos(col=2, row=3)

    fuse = TensorReshapeNode()
    fuse.inputs["op"].default_value  = "cat"
    fuse.inputs["dim"].default_value = 1
    graph.add_node(fuse); positions[fuse.id] = pos(col=3, row=2)

    head = _lin(2); graph.add_node(head); positions[head.id] = pos(col=4, row=2)

    loss = LossComputeNode(); loss.inputs["loss_type"].default_value="cross_entropy"
    graph.add_node(loss); positions[loss.id] = pos(col=5, row=2)

    data_out = TrainMarkerNode(); data_out.inputs["kind"].default_value="loss"
    graph.add_node(data_out); positions[data_out.id] = pos(col=6, row=2)

    graph.add_connection(audio_in.id,"tensor",a1.id,"tensor_in")
    graph.add_connection(image_in.id,"tensor",img_flat.id,"tensor_in"); graph.add_connection(img_flat.id,"tensor_out",i1.id,"tensor_in")
    graph.add_connection(a1.id,"tensor_out",fuse.id,"t1"); graph.add_connection(i1.id,"tensor_out",fuse.id,"t2")
    graph.add_connection(fuse.id,"tensor",head.id,"tensor_in")
    # (output port is `tensor` for both the legacy TensorCatNode and the
    # consolidated TensorReshapeNode, so the wire above stays valid.)
    graph.add_connection(head.id,"tensor_out",loss.id,"pred"); graph.add_connection(label_in.id,"tensor",loss.id,"target")
    graph.add_connection(loss.id,"loss",data_out.id,"tensor_in")
    return positions
