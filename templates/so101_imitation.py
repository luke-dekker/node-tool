"""SO-101 Imitation Learning — train ACT policy from demonstrations, deploy on hardware.

Two-row layout:
  Row 0 (Training):  LeRobot Dataset → ACT Policy (train mode) → MSE Loss → Train Output
  Row 1 (Deploy):    Camera → ACT Policy (inference mode) → Feetech Servo Bus

Showcases LeRobot + Robotics + PyTorch plugins working together.
"""
from __future__ import annotations
from core.graph import Graph
from templates._helpers import grid


LABEL = "SO-101 Imitation Learning"
DESCRIPTION = (
    "Train an ACT policy from LeRobot demonstrations, then deploy it on "
    "the SO-101 robot arm with camera input and servo control."
)


def build(graph: Graph) -> dict[str, tuple[int, int]]:
    from plugins.lerobot.nodes.dataset   import LeRobotDatasetNode
    from plugins.lerobot.nodes.policy    import ACTPolicyNode
    from plugins.lerobot.nodes.camera    import CameraInputNode
    from plugins.lerobot.nodes.servo_bus import FeetechServoBusNode
    from nodes.pytorch.train_output      import TrainOutputNode

    pos = grid(step_x=240, step_y=200)
    positions = {}

    # ── Row 0: Training pipeline ────────────────────────────────────────
    # Dataset (loads recorded demonstrations)
    ds = LeRobotDatasetNode()
    ds.inputs["repo_id"].default_value = "lerobot/so100_test"
    ds.inputs["batch_size"].default_value = 32
    ds.inputs["task_id"].default_value = "so101_train"
    graph.add_node(ds)
    positions[ds.id] = pos(0, 0)

    # ACT Policy (training — computes loss internally)
    act_train = ACTPolicyNode()
    act_train.inputs["observation_dim"].default_value = 6
    act_train.inputs["action_dim"].default_value = 6
    act_train.inputs["chunk_size"].default_value = 100
    act_train.inputs["hidden_dim"].default_value = 512
    act_train.inputs["mode"].default_value = "train"
    graph.add_node(act_train)
    positions[act_train.id] = pos(1, 0)

    # Train Output (loss_is_output: ACT loss feeds directly to backprop)
    tout = TrainOutputNode()
    tout.inputs["task_name"].default_value = "so101_train"
    tout.inputs["loss_is_output"].default_value = True
    graph.add_node(tout)
    positions[tout.id] = pos(2, 0)

    # Wire training row: dataset → ACT → train output
    graph.add_connection(ds.id, "x", act_train.id, "observation")
    graph.add_connection(ds.id, "label", act_train.id, "action")
    graph.add_connection(act_train.id, "loss", tout.id, "tensor_in")

    # ── Row 1: Deployment pipeline ──────────────────────────────────────
    # Camera
    cam = CameraInputNode()
    cam.inputs["camera_id"].default_value = 0
    cam.inputs["width"].default_value = 320
    cam.inputs["height"].default_value = 240
    graph.add_node(cam)
    positions[cam.id] = pos(0, 1)

    # ACT Policy (inference) — shares weights after training
    act_deploy = ACTPolicyNode()
    act_deploy.inputs["observation_dim"].default_value = 6
    act_deploy.inputs["action_dim"].default_value = 6
    act_deploy.inputs["chunk_size"].default_value = 100
    act_deploy.inputs["hidden_dim"].default_value = 512
    act_deploy.inputs["mode"].default_value = "inference"
    graph.add_node(act_deploy)
    positions[act_deploy.id] = pos(1, 1)

    # Servo Bus
    servos = FeetechServoBusNode()
    servos.inputs["port"].default_value = ""
    servos.inputs["write"].default_value = False
    graph.add_node(servos)
    positions[servos.id] = pos(2, 1)

    # Wire deployment row
    # Camera image → ACT observation (in practice you'd add a vision encoder;
    # for this template we show the data flow concept)
    graph.add_connection(act_deploy.id, "predicted_action", servos.id, "target_joints")

    return positions
