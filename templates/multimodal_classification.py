"""Multi-modal classification — audio + image fusion with loss-as-output."""
from __future__ import annotations
from core.graph import Graph
from templates._helpers import grid

LABEL = "Multi-Modal Classification"
DESCRIPTION = "Audio + image fusion with per-modality encoders, concat, loss-as-output."

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
    _ensure_data()
    from nodes.pytorch.dataset       import DatasetNode
    from nodes.pytorch.linear        import LinearNode
    from nodes.pytorch.flatten       import FlattenNode
    from nodes.pytorch.tensor_cat    import TensorCatNode
    from nodes.pytorch.loss_compute  import LossComputeNode
    from nodes.pytorch.train_output  import TrainOutputNode

    pos = grid(step_x=220); positions = {}

    ds = DatasetNode(); ds.inputs["path"].default_value=DEMO_DIR; ds.inputs["batch_size"].default_value=8
    graph.add_node(ds); positions[ds.id] = pos(col=0, row=2)

    a1 = LinearNode(); a1.inputs["in_features"].default_value=64; a1.inputs["out_features"].default_value=16; a1.inputs["activation"].default_value="relu"
    graph.add_node(a1); positions[a1.id] = pos(col=1, row=1)

    img_flat = FlattenNode(); graph.add_node(img_flat); positions[img_flat.id] = pos(col=1, row=3)
    i1 = LinearNode(); i1.inputs["in_features"].default_value=192; i1.inputs["out_features"].default_value=16; i1.inputs["activation"].default_value="relu"
    graph.add_node(i1); positions[i1.id] = pos(col=2, row=3)

    fuse = TensorCatNode(); fuse.inputs["dim"].default_value=1
    graph.add_node(fuse); positions[fuse.id] = pos(col=3, row=2)

    head = LinearNode(); head.inputs["in_features"].default_value=32; head.inputs["out_features"].default_value=2
    graph.add_node(head); positions[head.id] = pos(col=4, row=2)

    loss = LossComputeNode(); loss.inputs["loss_type"].default_value="cross_entropy"
    graph.add_node(loss); positions[loss.id] = pos(col=5, row=2)

    target = TrainOutputNode(); target.inputs["loss_is_output"].default_value=True
    graph.add_node(target); positions[target.id] = pos(col=6, row=2)

    graph.add_connection(ds.id,"audio",a1.id,"tensor_in")
    graph.add_connection(ds.id,"image",img_flat.id,"tensor_in"); graph.add_connection(img_flat.id,"tensor_out",i1.id,"tensor_in")
    graph.add_connection(a1.id,"tensor_out",fuse.id,"t1"); graph.add_connection(i1.id,"tensor_out",fuse.id,"t2")
    graph.add_connection(fuse.id,"tensor",head.id,"tensor_in")
    graph.add_connection(head.id,"tensor_out",loss.id,"pred"); graph.add_connection(ds.id,"label",loss.id,"target")
    graph.add_connection(loss.id,"loss",target.id,"tensor_in")
    return positions
