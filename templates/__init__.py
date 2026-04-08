"""Prebuilt graph templates.

Each template is a `build(graph)` function that:
  - instantiates the nodes it needs
  - sets sensible default values
  - adds nodes and connections to the given Graph
  - returns a dict mapping node.id -> (x, y) for canvas layout

The GUI's File -> Templates menu is populated from the TEMPLATES list below.
Adding a new template is two lines: write a builder file, append the entry here.
"""
from __future__ import annotations
from typing import Callable
from core.graph import Graph

# Type alias for clarity
TemplateBuilder = Callable[[Graph], dict[str, tuple[int, int]]]

from templates.mnist_mlp           import build as build_mnist_mlp
from templates.mnist_cnn           import build as build_mnist_cnn
from templates.mnist_vae           import build as build_mnist_vae
from templates.transfer_learning   import build as build_transfer_learning
from templates.csv_quick_look      import build as build_csv_quick_look
from templates.csv_cleaning        import build as build_csv_cleaning
from templates.csv_join_aggregate  import build as build_csv_join
from templates.csv_sklearn         import build as build_csv_sklearn
from templates.kmeans_pca          import build as build_kmeans_pca
from templates.time_series_lstm    import build as build_time_series_lstm

# (label, description, builder)
TEMPLATES: list[tuple[str, str, TemplateBuilder]] = [
    ("MNIST Classifier (MLP)",
     "Hello-world MLP on MNIST. Flatten -> Linear+ReLU -> Linear -> CrossEntropy.",
     build_mnist_mlp),

    ("MNIST Classifier (CNN)",
     "Convolutional pipeline on MNIST. Conv -> Pool -> Conv -> Pool -> Linear.",
     build_mnist_cnn),

    ("MNIST VAE (Image Generator)",
     "Variational autoencoder on MNIST. Trains a generator over the latent space.",
     build_mnist_vae),

    ("Transfer Learning (ResNet18)",
     "Pretrained ResNet18, frozen backbone, new classification head on CIFAR-10.",
     build_transfer_learning),

    ("CSV Quick Look",
     "Load a CSV and show shape, info, describe, head. The 'what's in this file' workflow.",
     build_csv_quick_look),

    ("CSV Cleaning Pipeline",
     "Real ETL: load -> drop NA -> fill NA -> normalize -> sort -> select cols.",
     build_csv_cleaning),

    ("Two-Table Join + Aggregate",
     "Load two CSVs -> merge -> filter -> groupby -> describe.",
     build_csv_join),

    ("CSV -> Sklearn Regression",
     "End-to-end tabular ML. Load -> clean -> split -> scale -> fit -> predict -> R2.",
     build_csv_sklearn),

    ("K-Means + PCA Visualization",
     "Unsupervised pipeline. Standardize -> KMeans -> PCA to 2D for inspection.",
     build_kmeans_pca),

    ("Time Series Forecasting (LSTM)",
     "Synthetic sine wave forecasting with LSTM. No external data needed.",
     build_time_series_lstm),
]
