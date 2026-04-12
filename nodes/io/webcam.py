"""Webcam capture node — capture video frames from any USB camera via OpenCV."""
from __future__ import annotations
from typing import Any
from core.node import BaseNode, PortType


class WebcamCaptureNode(BaseNode):
    """Capture frames from a USB webcam. Outputs image tensor + raw ndarray."""
    type_name   = "io_webcam"
    label       = "Webcam Capture"
    category    = "IO"
    subcategory = "Camera"
    description = (
        "Capture frames from a USB camera. Outputs an image tensor "
        "(C, H, W) float in [0, 1] and a raw (H, W, 3) uint8 ndarray. "
        "Uses OpenCV (pip install opencv-python)."
    )

    def __init__(self):
        self._cap = None
        self._cap_id = -1
        super().__init__()

    def _setup_ports(self) -> None:
        self.add_input("camera_id",  PortType.INT, 0,
                       description="Camera device index (0 = default webcam)")
        self.add_input("width",      PortType.INT, 320)
        self.add_input("height",     PortType.INT, 240)
        self.add_input("enabled",    PortType.BOOL, True)
        self.add_output("image",     PortType.TENSOR,
                        description="(3, H, W) float tensor in [0, 1]")
        self.add_output("raw_image", PortType.IMAGE,
                        description="(H, W, 3) uint8 ndarray for viz")
        self.add_output("info",      PortType.STRING)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        cam_id  = int(inputs.get("camera_id") or 0)
        width   = int(inputs.get("width") or 320)
        height  = int(inputs.get("height") or 240)
        enabled = bool(inputs.get("enabled", True))
        empty = {"image": None, "raw_image": None, "info": ""}

        if not enabled:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            return {**empty, "info": "Camera disabled"}

        try:
            import cv2
            import numpy as np
            import torch

            if self._cap is None or self._cap_id != cam_id:
                if self._cap is not None:
                    self._cap.release()
                self._cap = cv2.VideoCapture(cam_id)
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                self._cap_id = cam_id

            ret, frame = self._cap.read()
            if not ret or frame is None:
                return {**empty, "info": f"Camera {cam_id}: no frame"}

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frame_rgb.shape[:2] != (height, width):
                frame_rgb = cv2.resize(frame_rgb, (width, height))

            tensor = torch.from_numpy(frame_rgb.astype(np.float32) / 255.0)
            tensor = tensor.permute(2, 0, 1).contiguous()

            return {
                "image": tensor,
                "raw_image": frame_rgb,
                "info": f"Camera {cam_id}: {width}x{height}",
            }
        except ImportError:
            return {**empty, "info": "OpenCV not installed. Run: pip install opencv-python"}
        except Exception as exc:
            return {**empty, "info": f"Camera error: {exc}"}

    def export(self, iv, ov):
        return ["import cv2", "import torch", "import numpy as np"], [
            f"_cap = cv2.VideoCapture({self._val(iv, 'camera_id')})",
            f"_, _frame = _cap.read()",
            f"{ov.get('image', '_cam_img')} = torch.from_numpy("
            f"cv2.cvtColor(_frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0"
            f").permute(2, 0, 1)",
        ]
