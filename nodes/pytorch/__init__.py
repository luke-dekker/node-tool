"""PyTorch nodes — imports each node class from its own individual file.

Obsolete nodes (BatchInput, TrainingConfig, individual dataset loaders,
MultimodalModel, etc.) have been DELETED — not just hidden. The universal
DatasetNode, TrainOutputNode, and panel-driven training replace them all.
"""

# Layers — LayerNode absorbs 13 single-in single-out layer types via `kind`
# dropdown (linear, conv2d, batchnorm{1d,2d}, layernorm, dropout, embedding,
# activation, positional_encoding, transformer_encoder, max_pool2d, avg_pool2d,
# adaptive_avg_pool2d). Old class names are aliased so imports keep working;
# callers must set `kind` on the instance to recover specific behavior.
from nodes.pytorch.layer import LayerNode
LinearNode                  = LayerNode  # set kind="linear" (default)
Conv2dNode                  = LayerNode  # set kind="conv2d"
BatchNorm1dNode             = LayerNode  # set kind="batchnorm1d"
BatchNorm2dNode             = LayerNode  # set kind="batchnorm2d"
LayerNormNode               = LayerNode  # set kind="layernorm"
DropoutNode                 = LayerNode  # set kind="dropout"
EmbeddingNode               = LayerNode  # set kind="embedding"
ActivationNode              = LayerNode  # set kind="activation"
PositionalEncodingNode      = LayerNode  # set kind="positional_encoding"
TransformerEncoderLayerNode = LayerNode  # set kind="transformer_encoder"
Pool2dNode                  = LayerNode  # set kind="max_pool2d" / "avg_pool2d" / "adaptive_avg_pool2d"
MaxPool2dNode               = LayerNode  # set kind="max_pool2d"
AvgPool2dNode               = LayerNode  # set kind="avg_pool2d"
AdaptiveAvgPool2dNode       = LayerNode  # set kind="adaptive_avg_pool2d"

from nodes.pytorch.flatten import FlattenNode
from nodes.pytorch.multihead_attention import MultiheadAttentionNode

# Losses
from nodes.pytorch.mse_loss import LossFnNode
from nodes.pytorch.loss_compute import LossComputeNode

# Optimizers
from nodes.pytorch.adam import OptimizerNode

# Schedulers
from nodes.pytorch.step_lr import LRSchedulerNode

# Training
from nodes.pytorch.input_marker import InputMarkerNode
from nodes.pytorch.train_marker import TrainMarkerNode

# Datasets
from nodes.pytorch.streaming_buffer import StreamingBufferNode

# Dataset transforms
from nodes.pytorch.apply_transform import ApplyTransformNode
from nodes.pytorch.train_val_split import TrainValSplitNode
from nodes.pytorch.train_val_test_split import TrainValTestSplitNode
from nodes.pytorch.compose_transforms import ComposeTransformsNode
# Image transforms — ImageTransformNode replaces 9 per-op nodes (resize,
# normalize, center_crop, random_crop, h_flip, v_flip, grayscale,
# color_jitter, to_tensor). Audio (mel_spectrogram) and NLP (hf_tokenizer)
# transforms stay separate because their domains differ.
from nodes.pytorch.image_transform import ImageTransformNode
from nodes.pytorch.mel_spectrogram import MelSpectrogramTransformNode
from nodes.pytorch.hf_tokenizer_transform import HFTokenizerTransformNode
from nodes.pytorch.char_tokenizer import CharTokenizerNode
from nodes.pytorch.audio_pad_collate import AudioPadCollateNode

# Tensor data / ops — consolidated 13 → 5 (with PrintTensor as side-effect node).
# TensorCreateNode now absorbs from_list (kind="from_list").
# TensorOpNode absorbs binary (add/sub/mul/div), argmax, softmax, einsum, mux.
# TensorReshapeNode absorbs cat, stack, split, shape_op (reshape/squeeze/unsqueeze/permute/transpose).
from nodes.pytorch.tensor_create  import TensorCreateNode
from nodes.pytorch.tensor_op      import TensorOpNode
from nodes.pytorch.tensor_reshape import TensorReshapeNode
from nodes.pytorch.tensor_info    import TensorInfoNode
from nodes.pytorch.print_tensor   import PrintTensorNode

# Back-compat aliases
TensorFromListNode  = TensorCreateNode  # set fill="from_list"
TensorAddNode       = TensorOpNode      # set op="add"
TensorMulNode       = TensorOpNode      # set op="mul"
TensorBinaryOpNode  = TensorOpNode
ArgmaxNode          = TensorOpNode      # set op="argmax"
SoftmaxOpNode       = TensorOpNode      # set op="softmax"
TensorEinsumNode    = TensorOpNode      # set op="einsum"
TensorMuxNode       = TensorOpNode      # set op="mux"
TensorCatNode       = TensorReshapeNode  # set op="cat"
TensorStackNode     = TensorReshapeNode  # set op="stack"
TensorSplitNode     = TensorReshapeNode  # set op="split"
TensorShapeOpNode   = TensorReshapeNode  # set op="reshape"

# Recurrent — RecurrentLayerNode absorbs RNN/LSTM/GRU via `kind` dropdown.
# `cell` output is populated only when kind=lstm.
from nodes.pytorch.recurrent_layer import RecurrentLayerNode
RNNNode  = RecurrentLayerNode  # set kind="rnn"
LSTMNode = RecurrentLayerNode  # set kind="lstm" (default)
GRUNode  = RecurrentLayerNode  # set kind="gru"
from nodes.pytorch.pack_sequence import PackSequenceNode
from nodes.pytorch.unpack_sequence import UnpackSequenceNode
from nodes.pytorch.reshape_for_loss import ReshapeForLossNode

# Architecture — BranchOpNode absorbs AddBranches + ConcatBranches (op dropdown)
from nodes.pytorch.residual_block import ResidualBlockNode
from nodes.pytorch.branch_op      import BranchOpNode
from nodes.pytorch.custom_module  import CustomModuleNode
ConcatBranchesNode = BranchOpNode  # set op="concat"
AddBranchesNode    = BranchOpNode  # set op="add"

# Models / persistence — ModelIONode absorbs Save/Load weights, Load Model, Export ONNX
from nodes.pytorch.model_io  import ModelIONode
from nodes.pytorch.apply_module import ApplyModuleNode
from nodes.pytorch.gate import GateNode
from nodes.pytorch.class_module_import import ClassModuleImportNode
SaveWeightsNode = ModelIONode  # set mode="save_weights" (default)
LoadWeightsNode = ModelIONode  # set mode="load_into"
ExportONNXNode  = ModelIONode  # set mode="export_onnx"
LoadModelNode   = ModelIONode  # set mode="load_full"

# Backbones
from nodes.pytorch.pretrained_backbone import PretrainedBackboneNode
# FreezeLayersNode now absorbs FreezeNamedLayersNode (via mode="by_name").
# Aliases below keep old imports working.
from nodes.pytorch.freeze_backbone import (
    FreezeLayersNode, FreezeBackboneNode, FreezeNamedLayersNode,
)
from nodes.pytorch.model_info_persist import ModelInfoNode

# VAE/AE — LatentNode absorbs Reparameterize + LatentSampler;
# VAELossNode absorbs KLDivergence (mode="kl") + the original combine logic
from nodes.pytorch.latent   import LatentNode
from nodes.pytorch.vae_loss import VAELossNode
ReparameterizeNode = LatentNode    # set mode="reparameterize" (default)
LatentSamplerNode  = LatentNode    # set mode="sample"
KLDivergenceNode   = VAELossNode   # set mode="kl"

# Visualization
from nodes.pytorch.tensor_viz import TensorVizNode
from nodes.pytorch.viz_training_curve import PlotTrainingCurveNode
from nodes.pytorch.viz_weight_hist import WeightHistogramNode
