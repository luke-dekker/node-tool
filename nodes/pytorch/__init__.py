"""PyTorch nodes — imports each node class from its own individual file."""

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

# Losses
from nodes.pytorch.mse_loss import MSELossNode
from nodes.pytorch.cross_entropy_loss import CrossEntropyLossNode
from nodes.pytorch.bce_loss import BCELossNode
from nodes.pytorch.bce_logits_loss import BCEWithLogitsNode
from nodes.pytorch.l1_loss import L1LossNode

# Optimizers
from nodes.pytorch.adam import AdamNode
from nodes.pytorch.sgd import SGDNode
from nodes.pytorch.adamw import AdamWNode

# Schedulers
from nodes.pytorch.step_lr import StepLRNode
from nodes.pytorch.multistep_lr import MultiStepLRNode
from nodes.pytorch.exponential_lr import ExponentialLRNode
from nodes.pytorch.cosine_lr import CosineAnnealingLRNode
from nodes.pytorch.reduce_lr_plateau import ReduceLROnPlateauNode

# Training
from nodes.pytorch.forward_pass import ForwardPassNode
from nodes.pytorch.training_config import TrainingConfigNode

# Tensor data / ops
from nodes.pytorch.rand_tensor import RandTensorNode
from nodes.pytorch.zeros_tensor import ZerosTensorNode
from nodes.pytorch.ones_tensor import OnesTensorNode
from nodes.pytorch.tensor_from_list import TensorFromListNode
from nodes.pytorch.tensor_shape import TensorShapeNode
from nodes.pytorch.tensor_info import TensorInfoNode
from nodes.pytorch.tensor_add import TensorAddNode
from nodes.pytorch.tensor_mul import TensorMulNode
from nodes.pytorch.argmax import ArgmaxNode
from nodes.pytorch.softmax_op import SoftmaxOpNode
from nodes.pytorch.print_tensor import PrintTensorNode

# Tensor ops (cat/stack/split/reshape/squeeze/etc.)
from nodes.pytorch.tensor_cat import TensorCatNode
from nodes.pytorch.tensor_stack import TensorStackNode
from nodes.pytorch.tensor_split import TensorSplitNode
from nodes.pytorch.tensor_reshape import TensorReshapeNode
from nodes.pytorch.tensor_unsqueeze import TensorUnsqueezeNode
from nodes.pytorch.tensor_squeeze import TensorSqueezeNode
from nodes.pytorch.tensor_transpose import TensorTransposeNode
from nodes.pytorch.tensor_permute import TensorPermuteNode
from nodes.pytorch.tensor_einsum import TensorEinsumNode

# Dataset ops
from nodes.pytorch.apply_transform import ApplyTransformNode
from nodes.pytorch.train_val_split import TrainValSplitNode
from nodes.pytorch.train_val_test_split import TrainValTestSplitNode
from nodes.pytorch.dataloader import DataLoaderNode
from nodes.pytorch.dataset_info import DatasetInfoNode

# Dataset sources
from nodes.pytorch.mnist_dataset import MNISTDatasetNode
from nodes.pytorch.cifar10_dataset import CIFAR10DatasetNode
from nodes.pytorch.dataloader_info import DataLoaderInfoNode
from nodes.pytorch.sample_batch import SampleBatchNode
from nodes.pytorch.csv_dataset import CSVDatasetNode
from nodes.pytorch.numpy_dataset import NumpyDatasetNode
from nodes.pytorch.image_folder_dataset import ImageFolderDatasetNode
from nodes.pytorch.hf_dataset import HuggingFaceDatasetNode
from nodes.pytorch.audio_folder_dataset import AudioFolderDatasetNode

# Dataset transforms
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

# Recurrent
from nodes.pytorch.rnn_layer import RNNLayerNode
from nodes.pytorch.gru_layer import GRULayerNode
from nodes.pytorch.lstm_layer import LSTMLayerNode
from nodes.pytorch.rnn_forward import RNNForwardNode
from nodes.pytorch.lstm_forward import LSTMForwardNode
from nodes.pytorch.pack_sequence import PackSequenceNode
from nodes.pytorch.unpack_sequence import UnpackSequenceNode

# Architecture
from nodes.pytorch.residual_block import ResidualBlockNode
from nodes.pytorch.concat_branches import ConcatBranchesNode
from nodes.pytorch.add_branches import AddBranchesNode
from nodes.pytorch.custom_module import CustomModuleNode
from nodes.pytorch.multimodal_model import MultimodalModelNode
from nodes.pytorch.folder_multimodal_dataset import FolderMultimodalDatasetNode
from nodes.pytorch.multi_dataset import MultiDatasetNode
from nodes.pytorch.multimodal_training_config import MultimodalTrainingConfigNode
from nodes.pytorch.batch_input import BatchInputNode

# Persistence / Save & Load
from nodes.pytorch.save_weights import SaveWeightsNode
from nodes.pytorch.load_weights import LoadWeightsNode
from nodes.pytorch.save_checkpoint import SaveCheckpointNode
from nodes.pytorch.load_checkpoint import LoadCheckpointNode
from nodes.pytorch.export_onnx import ExportONNXNode
from nodes.pytorch.save_full_model import SaveFullModelNode
from nodes.pytorch.pretrained_block import PretrainedBlockNode
from nodes.pytorch.load_model import LoadModelNode
from nodes.pytorch.apply_module import ApplyModuleNode
from nodes.pytorch.train_output import TrainOutputNode
from nodes.pytorch.dataset import DatasetNode
from nodes.pytorch.gate import GateNode
from nodes.pytorch.class_module_import import ClassModuleImportNode
from nodes.pytorch.model_info_persist import ModelInfoPersistNode

# Backbones
from nodes.pytorch.resnet18 import ResNet18Node
from nodes.pytorch.resnet50 import ResNet50Node
from nodes.pytorch.mobilenet_v3 import MobileNetV3Node
from nodes.pytorch.efficientnet_b0 import EfficientNetB0Node
from nodes.pytorch.freeze_backbone import FreezeLayersNode, FreezeBackboneNode
from nodes.pytorch.reshape_for_loss import ReshapeForLossNode
from nodes.pytorch.text_dataset import TextDatasetNode
from nodes.pytorch.model_info import ModelInfoNode
from nodes.pytorch.freeze_named_layers import FreezeNamedLayersNode

# Autoencoder building blocks (per-layer composition — no monolithic VAE/AE node)
# VAENode and AutoencoderNode were deleted in favor of building VAEs and AEs
# from explicit Linear/Activation/Reparameterize chains in templates. The
# pieces below are the per-layer primitives that the templates compose.
from nodes.pytorch.reparameterize import ReparameterizeNode
from nodes.pytorch.kl_divergence import KLDivergenceNode
from nodes.pytorch.vae_loss import VAELossNode
from nodes.pytorch.latent_sampler import LatentSamplerNode
from nodes.pytorch.gaussian_noise import GaussianNoiseNode

# Generic loss-as-output node — used with TrainingConfig.loss_is_output=True
from nodes.pytorch.loss_compute import LossComputeNode

# Visualization
from nodes.pytorch.viz_tensor import PlotTensorNode
from nodes.pytorch.viz_training_curve import PlotTrainingCurveNode
from nodes.pytorch.viz_tensor_hist import TensorHistogramNode
from nodes.pytorch.viz_tensor_scatter import TensorScatterNode
from nodes.pytorch.viz_show_image import ShowImageNode
from nodes.pytorch.viz_weight_hist import WeightHistogramNode
