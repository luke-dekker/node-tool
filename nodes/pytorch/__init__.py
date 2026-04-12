"""PyTorch nodes — imports each node class from its own individual file.

Obsolete nodes (BatchInput, TrainingConfig, individual dataset loaders,
MultimodalModel, etc.) have been DELETED — not just hidden. The universal
DatasetNode, TrainOutputNode, and panel-driven training replace them all.
"""

# Layers
from nodes.pytorch.flatten import FlattenNode
from nodes.pytorch.linear import LinearNode
from nodes.pytorch.dropout import DropoutNode
from nodes.pytorch.batchnorm1d import BatchNorm1dNode
from nodes.pytorch.embedding import EmbeddingNode
from nodes.pytorch.activation import ActivationNode
from nodes.pytorch.conv2d import Conv2dNode
from nodes.pytorch.maxpool2d import MaxPool2dNode
from nodes.pytorch.avgpool2d import AvgPool2dNode
from nodes.pytorch.batchnorm2d import BatchNorm2dNode
from nodes.pytorch.adaptive_avgpool2d import AdaptiveAvgPool2dNode
from nodes.pytorch.layer_norm import LayerNormNode
from nodes.pytorch.multihead_attention import MultiheadAttentionNode
from nodes.pytorch.transformer_encoder_layer import TransformerEncoderLayerNode
from nodes.pytorch.positional_encoding import PositionalEncodingNode

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
from nodes.pytorch.to_tensor_transform import ToTensorTransformNode
from nodes.pytorch.resize_transform import ResizeTransformNode
from nodes.pytorch.normalize_transform import NormalizeTransformNode
from nodes.pytorch.random_hflip import RandomHFlipTransformNode
from nodes.pytorch.random_vflip import RandomVFlipTransformNode
from nodes.pytorch.center_crop import CenterCropTransformNode
from nodes.pytorch.random_crop import RandomCropTransformNode
from nodes.pytorch.grayscale import GrayscaleTransformNode
from nodes.pytorch.color_jitter import ColorJitterTransformNode
from nodes.pytorch.mel_spectrogram import MelSpectrogramTransformNode
from nodes.pytorch.hf_tokenizer_transform import HFTokenizerTransformNode

# Tensor data / ops
from nodes.pytorch.tensor_create import TensorCreateNode
from nodes.pytorch.tensor_from_list import TensorFromListNode
from nodes.pytorch.tensor_shape import TensorShapeNode
from nodes.pytorch.tensor_info import TensorInfoNode
from nodes.pytorch.tensor_add import TensorAddNode
from nodes.pytorch.tensor_mul import TensorMulNode
from nodes.pytorch.argmax import ArgmaxNode
from nodes.pytorch.softmax_op import SoftmaxOpNode
from nodes.pytorch.print_tensor import PrintTensorNode
from nodes.pytorch.tensor_cat import TensorCatNode
from nodes.pytorch.tensor_stack import TensorStackNode
from nodes.pytorch.tensor_split import TensorSplitNode
from nodes.pytorch.tensor_shape_op import TensorShapeOpNode
from nodes.pytorch.tensor_transpose import TensorTransposeNode
from nodes.pytorch.tensor_permute import TensorPermuteNode
from nodes.pytorch.tensor_einsum import TensorEinsumNode
from nodes.pytorch.tensor_mux import TensorMuxNode

# Recurrent
from nodes.pytorch.rnn import RNNNode
from nodes.pytorch.lstm import LSTMNode
from nodes.pytorch.gru import GRUNode
from nodes.pytorch.pack_sequence import PackSequenceNode
from nodes.pytorch.unpack_sequence import UnpackSequenceNode
from nodes.pytorch.reshape_for_loss import ReshapeForLossNode

# Architecture
from nodes.pytorch.residual_block import ResidualBlockNode
from nodes.pytorch.concat_branches import ConcatBranchesNode
from nodes.pytorch.add_branches import AddBranchesNode
from nodes.pytorch.custom_module import CustomModuleNode

# Models / persistence
from nodes.pytorch.save_weights import SaveWeightsNode
from nodes.pytorch.load_weights import LoadWeightsNode
from nodes.pytorch.save_checkpoint import SaveCheckpointNode
from nodes.pytorch.load_checkpoint import LoadCheckpointNode
from nodes.pytorch.export_onnx import ExportONNXNode
from nodes.pytorch.save_full_model import SaveFullModelNode
from nodes.pytorch.pretrained_block import PretrainedBlockNode
from nodes.pytorch.load_model import LoadModelNode
from nodes.pytorch.apply_module import ApplyModuleNode
from nodes.pytorch.gate import GateNode
from nodes.pytorch.class_module_import import ClassModuleImportNode

# Backbones
from nodes.pytorch.pretrained_backbone import PretrainedBackboneNode
from nodes.pytorch.freeze_backbone import FreezeLayersNode, FreezeBackboneNode
from nodes.pytorch.model_info_persist import ModelInfoNode
from nodes.pytorch.freeze_named_layers import FreezeNamedLayersNode

# VAE/AE per-layer building blocks
from nodes.pytorch.reparameterize import ReparameterizeNode
from nodes.pytorch.kl_divergence import KLDivergenceNode
from nodes.pytorch.vae_loss import VAELossNode
from nodes.pytorch.latent_sampler import LatentSamplerNode

# Visualization
from nodes.pytorch.tensor_viz import TensorVizNode
from nodes.pytorch.viz_training_curve import PlotTrainingCurveNode
from nodes.pytorch.viz_weight_hist import WeightHistogramNode
