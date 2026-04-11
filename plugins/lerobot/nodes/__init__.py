"""LeRobot integration nodes — servo, policy, camera.

Dataset loading is handled by the universal DatasetNode (set path to a
HuggingFace repo ID like "lerobot/so100_test" or a local folder).
"""
from plugins.lerobot.nodes.servo_bus import FeetechServoBusNode
from plugins.lerobot.nodes.policy import ACTPolicyNode
from plugins.lerobot.nodes.camera import CameraInputNode
