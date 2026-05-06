"""Consolidated torchvision image transform node.

Replaces nine per-op nodes with a single mode-dispatched node:
  to_tensor, resize, normalize, center_crop, random_crop,
  h_flip, v_flip, grayscale, color_jitter

Pick a `kind`; only the inputs that kind reads are consulted, the rest
are ignored. Output is always a torchvision Transform on the `transform`
port — feed it directly to ApplyTransform / ComposeTransforms.

Audio (mel_spectrogram) and NLP (hf_tokenizer) transforms stay separate
because their input/output domains differ from image transforms.
"""
from __future__ import annotations
from core.node import BaseNode, PortType


_KINDS = ["to_tensor", "resize", "normalize", "center_crop", "random_crop",
          "h_flip", "v_flip", "grayscale", "color_jitter"]


class ImageTransformNode(BaseNode):
    type_name   = "pt_image_transform"
    label       = "Image Transform"
    category    = "Data"
    subcategory = "Transforms"
    description = (
        "Build a torchvision image transform. Pick `kind`:\n"
        "  to_tensor    — no params\n"
        "  resize       — height, width\n"
        "  normalize    — mean, std (comma-separated per channel)\n"
        "  center_crop  — size\n"
        "  random_crop  — size, padding\n"
        "  h_flip       — p\n"
        "  v_flip       — p\n"
        "  grayscale    — num_channels (1 or 3)\n"
        "  color_jitter — brightness, contrast, saturation, hue"
    )

    def _setup_ports(self):
        self.add_input("kind",        PortType.STRING, "to_tensor", choices=_KINDS)
        # union of params across all kinds
        self.add_input("height",      PortType.INT,    224,    optional=True)
        self.add_input("width",       PortType.INT,    224,    optional=True)
        self.add_input("size",        PortType.INT,    224,    optional=True)
        self.add_input("padding",     PortType.INT,    4,      optional=True)
        self.add_input("p",           PortType.FLOAT,  0.5,    optional=True)
        self.add_input("num_channels",PortType.INT,    1,      optional=True)
        self.add_input("mean",        PortType.STRING, "0.5,0.5,0.5", optional=True)
        self.add_input("std",         PortType.STRING, "0.5,0.5,0.5", optional=True)
        self.add_input("brightness",  PortType.FLOAT,  0.2, optional=True)
        self.add_input("contrast",    PortType.FLOAT,  0.2, optional=True)
        self.add_input("saturation",  PortType.FLOAT,  0.2, optional=True)
        self.add_input("hue",         PortType.FLOAT,  0.0, optional=True)
        self.add_output("transform", PortType.TRANSFORM)

    def relevant_inputs(self, values):
        kind = (values.get("kind") or "to_tensor").strip()
        per_kind = {
            "to_tensor":    [],
            "resize":       ["height", "width"],
            "normalize":    ["mean", "std"],
            "center_crop":  ["size"],
            "random_crop":  ["size", "padding"],
            "h_flip":       ["p"],
            "v_flip":       ["p"],
            "grayscale":    ["num_channels"],
            "color_jitter": ["brightness", "contrast", "saturation", "hue"],
        }
        return ["kind"] + per_kind.get(kind, [])

    def execute(self, inputs):
        try:
            import torchvision.transforms as T
            kind = (inputs.get("kind") or "to_tensor").strip()
            if kind == "to_tensor":
                return {"transform": T.ToTensor()}
            if kind == "resize":
                h = int(inputs.get("height") or 224)
                w = int(inputs.get("width")  or 224)
                return {"transform": T.Resize((h, w))}
            if kind == "normalize":
                mean = [float(x) for x in str(inputs.get("mean") or "0.5").split(",")]
                std  = [float(x) for x in str(inputs.get("std")  or "0.5").split(",")]
                return {"transform": T.Normalize(mean=mean, std=std)}
            if kind == "center_crop":
                return {"transform": T.CenterCrop(int(inputs.get("size") or 224))}
            if kind == "random_crop":
                return {"transform": T.RandomCrop(int(inputs.get("size") or 32),
                                                  padding=int(inputs.get("padding") or 0))}
            if kind == "h_flip":
                return {"transform": T.RandomHorizontalFlip(p=float(inputs.get("p") or 0.5))}
            if kind == "v_flip":
                return {"transform": T.RandomVerticalFlip(p=float(inputs.get("p") or 0.5))}
            if kind == "grayscale":
                return {"transform": T.Grayscale(num_output_channels=int(inputs.get("num_channels") or 1))}
            if kind == "color_jitter":
                return {"transform": T.ColorJitter(
                    brightness=float(inputs.get("brightness") or 0),
                    contrast=float(inputs.get("contrast") or 0),
                    saturation=float(inputs.get("saturation") or 0),
                    hue=float(inputs.get("hue") or 0),
                )}
            return {"transform": None}
        except Exception:
            return {"transform": None}

    def export(self, iv, ov):
        out = ov["transform"]
        kind = (self.inputs["kind"].default_value or "to_tensor")
        if kind == "to_tensor":
            return ["from torchvision.transforms import ToTensor"], [f"{out} = ToTensor()"]
        if kind == "resize":
            h = self._val(iv, "height"); w = self._val(iv, "width")
            return ["from torchvision.transforms import Resize"], [f"{out} = Resize(({h}, {w}))"]
        if kind == "normalize":
            m = self._val(iv, "mean"); s = self._val(iv, "std")
            return ["from torchvision.transforms import Normalize"], [
                f"{out} = Normalize(mean=[float(x) for x in {m}.split(',')], std=[float(x) for x in {s}.split(',')])"
            ]
        if kind == "center_crop":
            s = self._val(iv, "size")
            return ["from torchvision.transforms import CenterCrop"], [f"{out} = CenterCrop({s})"]
        if kind == "random_crop":
            s = self._val(iv, "size"); p = self._val(iv, "padding")
            return ["from torchvision.transforms import RandomCrop"], [f"{out} = RandomCrop({s}, padding={p})"]
        if kind == "h_flip":
            p = self._val(iv, "p")
            return ["from torchvision.transforms import RandomHorizontalFlip"], [f"{out} = RandomHorizontalFlip(p={p})"]
        if kind == "v_flip":
            p = self._val(iv, "p")
            return ["from torchvision.transforms import RandomVerticalFlip"], [f"{out} = RandomVerticalFlip(p={p})"]
        if kind == "grayscale":
            c = self._val(iv, "num_channels")
            return ["from torchvision.transforms import Grayscale"], [f"{out} = Grayscale(num_output_channels={c})"]
        if kind == "color_jitter":
            b = self._val(iv, "brightness"); c = self._val(iv, "contrast")
            s = self._val(iv, "saturation"); h = self._val(iv, "hue")
            return ["from torchvision.transforms import ColorJitter"], [
                f"{out} = ColorJitter(brightness={b}, contrast={c}, saturation={s}, hue={h})"
            ]
        return [], [f"# unknown kind {kind!r}"]
