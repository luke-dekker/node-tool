"""Re-export shim — ImageTransformNode is the consolidated source of truth.

Old per-transform class names are kept as aliases so existing imports (tests,
saved graphs that pre-date the consolidation) keep working. The underlying
class IS ImageTransformNode — callers must set the appropriate `kind` (and
its kind-specific inputs) on the instance before execute(), since the alias
only fixes the name, not the default.

Audio (mel_spectrogram) and NLP (hf_tokenizer) transforms stay separate.
"""
from nodes.pytorch.image_transform        import ImageTransformNode
from nodes.pytorch.compose_transforms     import ComposeTransformsNode
from nodes.pytorch.apply_transform        import ApplyTransformNode
from nodes.pytorch.mel_spectrogram        import MelSpectrogramTransformNode
from nodes.pytorch.hf_tokenizer_transform import HFTokenizerTransformNode

# Back-compat aliases for the nine collapsed transforms. Same class — set
# the `kind` input to recover per-transform behavior.
ToTensorTransformNode      = ImageTransformNode  # kind="to_tensor"
ResizeTransformNode        = ImageTransformNode  # kind="resize"
NormalizeTransformNode     = ImageTransformNode  # kind="normalize"
CenterCropTransformNode    = ImageTransformNode  # kind="center_crop"
RandomCropTransformNode    = ImageTransformNode  # kind="random_crop"
RandomHFlipTransformNode   = ImageTransformNode  # kind="h_flip"
RandomVFlipTransformNode   = ImageTransformNode  # kind="v_flip"
GrayscaleTransformNode     = ImageTransformNode  # kind="grayscale"
ColorJitterTransformNode   = ImageTransformNode  # kind="color_jitter"

__all__ = [
    "ImageTransformNode",
    "ComposeTransformsNode", "ApplyTransformNode",
    "MelSpectrogramTransformNode", "HFTokenizerTransformNode",
    # back-compat
    "ToTensorTransformNode", "ResizeTransformNode", "NormalizeTransformNode",
    "CenterCropTransformNode", "RandomCropTransformNode",
    "RandomHFlipTransformNode", "RandomVFlipTransformNode",
    "GrayscaleTransformNode", "ColorJitterTransformNode",
]
