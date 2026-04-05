"""Tests for dataset pipeline nodes — no DPG, no network required."""
import pytest
import numpy as np


def test_numpy_dataset():
    from nodes.pytorch.dataset_sources import NumpyDatasetNode
    X = np.random.randn(100, 4).astype("float32")
    y = np.random.randint(0, 2, 100)
    result = NumpyDatasetNode().execute({"X": X, "y": y})
    ds = result["dataset"]
    assert ds is not None
    assert len(ds) == 100
    x0, y0 = ds[0]
    assert list(x0.shape) == [4]


def test_numpy_dataset_none_guard():
    from nodes.pytorch.dataset_sources import NumpyDatasetNode
    result = NumpyDatasetNode().execute({"X": None, "y": None})
    assert result["dataset"] is None


def test_csv_dataset(tmp_path):
    import pandas as pd
    from nodes.pytorch.dataset_sources import CSVDatasetNode
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "label": [0, 1, 0]})
    p = tmp_path / "test.csv"
    df.to_csv(p, index=False)
    result = CSVDatasetNode().execute({
        "file_path": str(p), "target_col": "label",
        "feature_cols": "", "normalize": False
    })
    assert result["dataset"] is not None
    assert len(result["dataset"]) == 3
    assert "3 samples" in result["info"]


def test_csv_dataset_normalize(tmp_path):
    import pandas as pd
    from nodes.pytorch.dataset_sources import CSVDatasetNode
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [0, 1, 0]})
    p = tmp_path / "norm.csv"
    df.to_csv(p, index=False)
    result = CSVDatasetNode().execute({
        "file_path": str(p), "target_col": "y",
        "feature_cols": "x", "normalize": True
    })
    assert result["dataset"] is not None


def test_to_tensor_transform():
    torchvision = pytest.importorskip("torchvision")
    from nodes.pytorch.dataset_transforms import ToTensorTransformNode
    result = ToTensorTransformNode().execute({})
    from torchvision.transforms import ToTensor
    assert isinstance(result["transform"], ToTensor)


def test_resize_transform():
    pytest.importorskip("torchvision")
    from nodes.pytorch.dataset_transforms import ResizeTransformNode
    result = ResizeTransformNode().execute({"height": 64, "width": 64})
    assert result["transform"] is not None


def test_normalize_transform():
    pytest.importorskip("torchvision")
    from nodes.pytorch.dataset_transforms import NormalizeTransformNode
    result = NormalizeTransformNode().execute({"mean": "0.5,0.5,0.5", "std": "0.5,0.5,0.5"})
    from torchvision.transforms import Normalize
    assert isinstance(result["transform"], Normalize)


def test_compose_transforms():
    pytest.importorskip("torchvision")
    from nodes.pytorch.dataset_transforms import ComposeTransformsNode, ToTensorTransformNode, ResizeTransformNode
    t1 = ResizeTransformNode().execute({"height": 32, "width": 32})["transform"]
    t2 = ToTensorTransformNode().execute({})["transform"]
    result = ComposeTransformsNode().execute({"t1": t1, "t2": t2, "t3": None, "t4": None, "t5": None, "t6": None})
    from torchvision.transforms import Compose
    assert isinstance(result["transform"], Compose)


def test_random_hflip_transform():
    pytest.importorskip("torchvision")
    from nodes.pytorch.dataset_transforms import RandomHFlipTransformNode
    result = RandomHFlipTransformNode().execute({"p": 0.5})
    assert result["transform"] is not None


def test_grayscale_transform():
    pytest.importorskip("torchvision")
    from nodes.pytorch.dataset_transforms import GrayscaleTransformNode
    result = GrayscaleTransformNode().execute({"num_channels": 1})
    assert result["transform"] is not None


def test_train_val_split():
    import numpy as np
    from nodes.pytorch.dataset_sources import NumpyDatasetNode
    from nodes.pytorch.dataset_ops import TrainValSplitNode
    X = np.random.randn(100, 4).astype("float32")
    y = np.random.randint(0, 2, 100)
    ds = NumpyDatasetNode().execute({"X": X, "y": y})["dataset"]
    result = TrainValSplitNode().execute({"dataset": ds, "val_ratio": 0.2, "seed": 42})
    assert result["train_dataset"] is not None
    assert result["val_dataset"] is not None
    assert len(result["train_dataset"]) == 80
    assert len(result["val_dataset"]) == 20
    assert "train=80" in result["info"]


def test_train_val_test_split():
    import numpy as np
    from nodes.pytorch.dataset_sources import NumpyDatasetNode
    from nodes.pytorch.dataset_ops import TrainValTestSplitNode
    X = np.random.randn(100, 4).astype("float32")
    y = np.random.randint(0, 2, 100)
    ds = NumpyDatasetNode().execute({"X": X, "y": y})["dataset"]
    result = TrainValTestSplitNode().execute({"dataset": ds, "val_ratio": 0.1, "test_ratio": 0.1, "seed": 42})
    assert len(result["train_dataset"]) == 80
    assert len(result["val_dataset"]) == 10
    assert len(result["test_dataset"]) == 10


def test_dataloader_node():
    import numpy as np
    from nodes.pytorch.dataset_sources import NumpyDatasetNode
    from nodes.pytorch.dataset_ops import DataLoaderNode
    X = np.random.randn(64, 4).astype("float32")
    y = np.random.randint(0, 2, 64)
    ds = NumpyDatasetNode().execute({"X": X, "y": y})["dataset"]
    result = DataLoaderNode().execute({"dataset": ds, "batch_size": 16, "shuffle": False,
                                       "num_workers": 0, "pin_memory": False, "drop_last": False})
    dl = result["dataloader"]
    assert dl is not None
    assert len(dl) == 4  # 64 / 16


def test_dataloader_none_guard():
    from nodes.pytorch.dataset_ops import DataLoaderNode
    result = DataLoaderNode().execute({"dataset": None, "batch_size": 32, "shuffle": True,
                                       "num_workers": 0, "pin_memory": False, "drop_last": False})
    assert result["dataloader"] is None


def test_apply_transform():
    import numpy as np
    from nodes.pytorch.dataset_sources import NumpyDatasetNode
    from nodes.pytorch.dataset_ops import ApplyTransformNode
    X = np.random.randn(10, 4).astype("float32")
    y = np.random.randint(0, 2, 10)
    ds = NumpyDatasetNode().execute({"X": X, "y": y})["dataset"]

    # Simple identity transform
    identity = lambda x: x * 2
    result = ApplyTransformNode().execute({"dataset": ds, "transform": identity})
    assert result["dataset"] is not None
    import torch
    x0_orig, _ = ds[0]
    x0_new, _ = result["dataset"][0]
    assert torch.allclose(x0_new, x0_orig * 2)


def test_dataset_info():
    import numpy as np
    from nodes.pytorch.dataset_sources import NumpyDatasetNode
    from nodes.pytorch.dataset_ops import DatasetInfoNode
    X = np.random.randn(50, 8).astype("float32")
    y = np.random.randint(0, 3, 50)
    ds = NumpyDatasetNode().execute({"X": X, "y": y})["dataset"]
    result = DatasetInfoNode().execute({"dataset": ds})
    assert "50" in result["info"]


def test_full_pipeline():
    """CSV -> TrainValSplit -> DataLoader -> usable for training."""
    import numpy as np, pandas as pd
    import tempfile, os
    from nodes.pytorch.dataset_sources import CSVDatasetNode
    from nodes.pytorch.dataset_ops import TrainValSplitNode, DataLoaderNode

    # Make temp CSV
    X = np.random.randn(200, 6).astype("float32")
    y = (X[:, 0] > 0).astype(int)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    df["label"] = y

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f, index=False)
        path = f.name

    try:
        ds = CSVDatasetNode().execute({"file_path": path, "target_col": "label",
                                        "feature_cols": "", "normalize": True})["dataset"]
        splits = TrainValSplitNode().execute({"dataset": ds, "val_ratio": 0.2, "seed": 0})
        train_dl = DataLoaderNode().execute({"dataset": splits["train_dataset"], "batch_size": 32,
                                              "shuffle": True, "num_workers": 0,
                                              "pin_memory": False, "drop_last": False})["dataloader"]
        val_dl = DataLoaderNode().execute({"dataset": splits["val_dataset"], "batch_size": 32,
                                            "shuffle": False, "num_workers": 0,
                                            "pin_memory": False, "drop_last": False})["dataloader"]
        assert train_dl is not None and val_dl is not None
        # Check a batch
        for xb, yb in train_dl:
            assert xb.shape[1] == 6
            break
    finally:
        os.unlink(path)


def test_registry_has_dataset_nodes():
    from nodes import NODE_REGISTRY
    expected = ["pt_csv_dataset", "pt_numpy_dataset", "pt_image_folder_dataset",
                "pt_hf_dataset", "pt_audio_folder_dataset",
                "pt_compose_transforms", "pt_to_tensor_transform", "pt_resize_transform",
                "pt_normalize_transform", "pt_train_val_split", "pt_train_val_test_split",
                "pt_dataloader", "pt_dataset_info", "pt_apply_transform"]
    for tn in expected:
        assert tn in NODE_REGISTRY, f"Missing from registry: {tn}"
