"""ImageInputNode — load an image file as a numpy ndarray (RGB uint8) or torch tensor.

The "drop an image and use it" entry point. Lets a user point at a JPG/PNG and
get either an `(H, W, 3)` uint8 ndarray (for viz / inspection) or a normalized
`(C, H, W)` float tensor (for ML pipelines).

Both outputs are populated on every execute — wire whichever the downstream
needs. The image is cached and only re-loaded when the path or normalize
setting changes.
"""
from __future__ import annotations
from typing import Any
from core.node import BaseNode, PortType


class ImageInputNode(BaseNode):
    type_name   = "image_input"
    label       = "Image Input"
    category    = "Python"
    subcategory = "Data"
    description = (
        "Load an image file from disk. Outputs both an (H, W, 3) uint8 ndarray "
        "and a normalized (C, H, W) float tensor — wire whichever you need. "
        "Re-loaded only when path or normalize changes."
    )

    def __init__(self):
        self._cached: dict | None = None
        self._cached_cfg: tuple = ()
        super().__init__()

    def _setup_ports(self) -> None:
        self.add_input("path",      PortType.STRING, default="image.png",
                       description="Path to a .png / .jpg / .bmp")
        self.add_input("normalize", PortType.BOOL,   default=True,
                       description="Scale tensor to [0, 1] floats")
        self.add_output("image",  PortType.IMAGE,
                        description="(H, W, 3) uint8 ndarray — for viz / inspection")
        self.add_output("tensor", PortType.TENSOR,
                        description="(C, H, W) float tensor — for ML pipelines")
        self.add_output("info",   PortType.STRING)

    def _load(self, path: str, normalize: bool) -> dict:
        from pathlib import Path
        import numpy as np
        import torch

        p = Path(path)
        if not p.exists():
            return {"image": None, "tensor": None,
                    "info": f"file not found: {path}"}

        # Try PIL first (handles png/jpg/bmp/etc), fall back to imageio
        try:
            from PIL import Image
            img = Image.open(p).convert("RGB")
            arr = np.asarray(img, dtype=np.uint8)  # (H, W, 3)
        except ImportError:
            try:
                import imageio.v3 as iio
                arr = iio.imread(p)
                if arr.ndim == 2:
                    arr = np.stack([arr] * 3, axis=-1)
                if arr.shape[-1] == 4:
                    arr = arr[..., :3]
                arr = arr.astype(np.uint8)
            except Exception as exc:
                return {"image": None, "tensor": None,
                        "info": f"load failed: {exc}"}
        except Exception as exc:
            return {"image": None, "tensor": None,
                    "info": f"load failed: {exc}"}

        # (H, W, 3) uint8 → (C, H, W) float
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        if normalize:
            tensor = tensor.float() / 255.0
        else:
            tensor = tensor.float()

        h, w, c = arr.shape
        info = f"{path}  ({h}x{w}x{c})  normalize={normalize}"
        return {"image": arr, "tensor": tensor, "info": info}

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        path      = str(inputs.get("path") or "image.png")
        normalize = bool(inputs.get("normalize", True))
        cfg = (path, normalize)
        if self._cached is None or self._cached_cfg != cfg:
            self._cached = self._load(path, normalize)
            self._cached_cfg = cfg
        return self._cached

    def export(self, iv, ov):
        path      = self._val(iv, "path")
        normalize = bool(self.inputs["normalize"].default_value)
        img_var    = ov.get("image",  "_img")
        tensor_var = ov.get("tensor", "_img_tensor")
        info_var   = ov.get("info",   "_img_info")
        return [
            "import numpy as np",
            "import torch",
            "from PIL import Image",
        ], [
            f"{img_var} = np.asarray(Image.open({path}).convert('RGB'), dtype=np.uint8)",
            f"{tensor_var} = torch.from_numpy({img_var}).permute(2, 0, 1).contiguous().float()"
            + (" / 255.0" if normalize else ""),
            f"{info_var} = f'{{{path}}}  {{{img_var}.shape}}  normalize={normalize}'",
        ]
