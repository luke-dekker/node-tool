"""Transform nodes — preprocessing for images, text, audio, tabular."""
from __future__ import annotations
from core.node import BaseNode
from core.node import PortType

CATEGORY = "Datasets"


class ComposeTransformsNode(BaseNode):
    type_name = "pt_compose_transforms"
    label = "Compose Transforms"
    category = CATEGORY
    description = "Chain up to 6 transforms into one."

    def _setup_ports(self):
        for i in range(1, 7):
            self.add_input(f"t{i}", PortType.TRANSFORM, default=None)
        self.add_output("transform", PortType.TRANSFORM)

    def execute(self, inputs):
        try:
            from torchvision.transforms import Compose
            ts = [inputs.get(f"t{i}") for i in range(1, 7) if inputs.get(f"t{i}") is not None]
            if not ts:
                return {"transform": None}
            return {"transform": Compose(ts)}
        except Exception:
            return {"transform": None}


class ToTensorTransformNode(BaseNode):
    type_name = "pt_to_tensor_transform"
    label = "To Tensor"
    category = CATEGORY
    description = "Convert PIL image or numpy array to a FloatTensor (torchvision.transforms.ToTensor)."

    def _setup_ports(self):
        self.add_output("transform", PortType.TRANSFORM)

    def execute(self, inputs):
        try:
            from torchvision.transforms import ToTensor
            return {"transform": ToTensor()}
        except Exception:
            return {"transform": None}


class ResizeTransformNode(BaseNode):
    type_name = "pt_resize_transform"
    label = "Resize"
    category = CATEGORY
    description = "Resize image to (height, width)."

    def _setup_ports(self):
        self.add_input("height", PortType.INT, default=224)
        self.add_input("width",  PortType.INT, default=224)
        self.add_output("transform", PortType.TRANSFORM)

    def execute(self, inputs):
        try:
            from torchvision.transforms import Resize
            h = int(inputs.get("height") or 224)
            w = int(inputs.get("width")  or 224)
            return {"transform": Resize((h, w))}
        except Exception:
            return {"transform": None}


class NormalizeTransformNode(BaseNode):
    type_name = "pt_normalize_transform"
    label = "Normalize"
    category = CATEGORY
    description = "Normalize tensor with mean and std. Enter comma-separated values per channel, e.g. '0.485,0.456,0.406'."

    def _setup_ports(self):
        self.add_input("mean", PortType.STRING, default="0.5,0.5,0.5")
        self.add_input("std",  PortType.STRING, default="0.5,0.5,0.5")
        self.add_output("transform", PortType.TRANSFORM)

    def execute(self, inputs):
        try:
            from torchvision.transforms import Normalize
            mean = [float(x) for x in str(inputs.get("mean") or "0.5").split(",")]
            std  = [float(x) for x in str(inputs.get("std")  or "0.5").split(",")]
            return {"transform": Normalize(mean=mean, std=std)}
        except Exception:
            return {"transform": None}


class RandomHFlipTransformNode(BaseNode):
    type_name = "pt_random_hflip_transform"
    label = "Random H Flip"
    category = CATEGORY
    description = "Randomly flip image horizontally with probability p."

    def _setup_ports(self):
        self.add_input("p", PortType.FLOAT, default=0.5)
        self.add_output("transform", PortType.TRANSFORM)

    def execute(self, inputs):
        try:
            from torchvision.transforms import RandomHorizontalFlip
            return {"transform": RandomHorizontalFlip(p=float(inputs.get("p") or 0.5))}
        except Exception:
            return {"transform": None}


class RandomVFlipTransformNode(BaseNode):
    type_name = "pt_random_vflip_transform"
    label = "Random V Flip"
    category = CATEGORY
    description = "Randomly flip image vertically with probability p."

    def _setup_ports(self):
        self.add_input("p", PortType.FLOAT, default=0.5)
        self.add_output("transform", PortType.TRANSFORM)

    def execute(self, inputs):
        try:
            from torchvision.transforms import RandomVerticalFlip
            return {"transform": RandomVerticalFlip(p=float(inputs.get("p") or 0.5))}
        except Exception:
            return {"transform": None}


class CenterCropTransformNode(BaseNode):
    type_name = "pt_center_crop_transform"
    label = "Center Crop"
    category = CATEGORY
    description = "Crop the center of an image to size x size."

    def _setup_ports(self):
        self.add_input("size", PortType.INT, default=224)
        self.add_output("transform", PortType.TRANSFORM)

    def execute(self, inputs):
        try:
            from torchvision.transforms import CenterCrop
            return {"transform": CenterCrop(int(inputs.get("size") or 224))}
        except Exception:
            return {"transform": None}


class RandomCropTransformNode(BaseNode):
    type_name = "pt_random_crop_transform"
    label = "Random Crop"
    category = CATEGORY
    description = "Randomly crop image to size with optional padding."

    def _setup_ports(self):
        self.add_input("size",    PortType.INT, default=32)
        self.add_input("padding", PortType.INT, default=4)
        self.add_output("transform", PortType.TRANSFORM)

    def execute(self, inputs):
        try:
            from torchvision.transforms import RandomCrop
            return {"transform": RandomCrop(int(inputs.get("size") or 32),
                                             padding=int(inputs.get("padding") or 0))}
        except Exception:
            return {"transform": None}


class GrayscaleTransformNode(BaseNode):
    type_name = "pt_grayscale_transform"
    label = "Grayscale"
    category = CATEGORY
    description = "Convert image to grayscale. num_output_channels: 1 or 3."

    def _setup_ports(self):
        self.add_input("num_channels", PortType.INT, default=1)
        self.add_output("transform", PortType.TRANSFORM)

    def execute(self, inputs):
        try:
            from torchvision.transforms import Grayscale
            return {"transform": Grayscale(num_output_channels=int(inputs.get("num_channels") or 1))}
        except Exception:
            return {"transform": None}


class ColorJitterTransformNode(BaseNode):
    type_name = "pt_color_jitter_transform"
    label = "Color Jitter"
    category = CATEGORY
    description = "Randomly change brightness, contrast, saturation, hue."

    def _setup_ports(self):
        self.add_input("brightness", PortType.FLOAT, default=0.2)
        self.add_input("contrast",   PortType.FLOAT, default=0.2)
        self.add_input("saturation", PortType.FLOAT, default=0.2)
        self.add_input("hue",        PortType.FLOAT, default=0.0)
        self.add_output("transform", PortType.TRANSFORM)

    def execute(self, inputs):
        try:
            from torchvision.transforms import ColorJitter
            return {"transform": ColorJitter(
                brightness=float(inputs.get("brightness") or 0),
                contrast=float(inputs.get("contrast") or 0),
                saturation=float(inputs.get("saturation") or 0),
                hue=float(inputs.get("hue") or 0),
            )}
        except Exception:
            return {"transform": None}


class MelSpectrogramTransformNode(BaseNode):
    type_name = "pt_mel_spectrogram_transform"
    label = "Mel Spectrogram"
    category = CATEGORY
    description = "Convert raw audio waveform to mel spectrogram (torchaudio)."

    def _setup_ports(self):
        self.add_input("sample_rate", PortType.INT, default=16000)
        self.add_input("n_mels",      PortType.INT, default=64)
        self.add_input("n_fft",       PortType.INT, default=400)
        self.add_output("transform", PortType.TRANSFORM)

    def execute(self, inputs):
        try:
            import torchaudio.transforms as T
            return {"transform": T.MelSpectrogram(
                sample_rate=int(inputs.get("sample_rate") or 16000),
                n_mels=int(inputs.get("n_mels") or 64),
                n_fft=int(inputs.get("n_fft") or 400),
            )}
        except Exception:
            return {"transform": None}


class HFTokenizerTransformNode(BaseNode):
    type_name = "pt_hf_tokenizer_transform"
    label = "HF Tokenizer"
    category = CATEGORY
    description = "Tokenize text using a HuggingFace tokenizer (e.g. 'bert-base-uncased'). Outputs a callable transform."

    def _setup_ports(self):
        self.add_input("model_name",  PortType.STRING, default="bert-base-uncased")
        self.add_input("max_length",  PortType.INT,    default=128)
        self.add_input("padding",     PortType.BOOL,   default=True)
        self.add_input("truncation",  PortType.BOOL,   default=True)
        self.add_output("transform", PortType.TRANSFORM)

    def execute(self, inputs):
        try:
            from transformers import AutoTokenizer
            name     = str(inputs.get("model_name") or "bert-base-uncased")
            max_len  = int(inputs.get("max_length") or 128)
            padding  = bool(inputs.get("padding", True))
            truncate = bool(inputs.get("truncation", True))
            tok = AutoTokenizer.from_pretrained(name)

            class TokenizerTransform:
                def __call__(self, text):
                    import torch
                    enc = tok(text, max_length=max_len, padding="max_length" if padding else False,
                              truncation=truncate, return_tensors="pt")
                    return {k: v.squeeze(0) for k, v in enc.items()}

            return {"transform": TokenizerTransform()}
        except Exception:
            return {"transform": None}


# Subcategory stamp
_SC = "Transforms"
from core.node import BaseNode as _BN
for _n, _c in list(globals().items()):
    if isinstance(_c, type) and issubclass(_c, _BN) and _c is not _BN:
        _c.subcategory = _SC
