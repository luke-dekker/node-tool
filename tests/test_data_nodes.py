"""Tests for numpy, pandas, sklearn, scipy, viz nodes — headless, no DPG."""
import pytest
import numpy as np

# ── NumPy ──────────────────────────────────────────────────────────────────

def test_np_arange():
    from nodes.numpy import NpArangeNode
    r = NpArangeNode().execute({"start": 0, "stop": 5, "step": 1})
    assert list(r["array"]) == [0,1,2,3,4]

def test_np_linspace():
    from nodes.numpy import NpLinspaceNode
    r = NpLinspaceNode().execute({"start": 0, "stop": 1, "num": 5})
    assert len(r["array"]) == 5
    assert abs(r["array"][-1] - 1.0) < 1e-6

def test_np_zeros():
    from nodes.numpy import NpZerosNode
    r = NpZerosNode().execute({"shape": "2,3"})
    assert r["array"].shape == (2, 3)

def test_np_reshape():
    from nodes.numpy import NpArangeNode, NpReshapeNode
    arr = NpArangeNode().execute({"start": 0, "stop": 6, "step": 1})["array"]
    r = NpReshapeNode().execute({"array": arr, "shape": "2,3"})
    assert r["result"].shape == (2, 3)

def test_np_dot():
    from nodes.numpy import NpDotNode
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    r = NpDotNode().execute({"a": a, "b": b})
    assert abs(r["result"] - 32.0) < 1e-6

def test_np_none_guard():
    from nodes.numpy import NpMeanNode
    r = NpMeanNode().execute({"array": None, "axis": -99})
    assert r["result"] is None

# ── Pandas ─────────────────────────────────────────────────────────────────

def test_pd_make_sample():
    from nodes.pandas import PdMakeSampleNode
    r = PdMakeSampleNode().execute({"rows": 50, "cols": 3, "seed": 0})
    import pandas as pd
    assert isinstance(r["df"], pd.DataFrame)
    assert r["df"].shape == (50, 4)  # 3 cols + label

def test_pd_filter():
    from nodes.pandas import PdMakeSampleNode, PdFilterRowsNode
    df = PdMakeSampleNode().execute({"rows": 100, "cols": 2, "seed": 42})["df"]
    r = PdFilterRowsNode().execute({"df": df, "column": "label", "op": "==", "value": 1.0})
    assert (r["result"]["label"] == 1).all()

def test_pd_to_numpy():
    from nodes.pandas import PdMakeSampleNode, PdToNumpyNode
    df = PdMakeSampleNode().execute({"rows": 10, "cols": 2, "seed": 0})["df"]
    r = PdToNumpyNode().execute({"df": df})
    assert r["array"].shape[0] == 10

def test_pd_xy_split():
    from nodes.pandas import PdMakeSampleNode, PdXYSplitNode
    df = PdMakeSampleNode().execute({"rows": 20, "cols": 3, "seed": 0})["df"]
    r = PdXYSplitNode().execute({"df": df, "label_col": "label"})
    assert "label" not in r["X"].columns
    assert r["y"].name == "label"

# ── Sklearn ────────────────────────────────────────────────────────────────

def test_sk_train_test_split():
    from nodes.pandas import PdMakeSampleNode, PdXYSplitNode, PdToNumpyNode
    from nodes.sklearn import SkTrainTestSplitNode
    df = PdMakeSampleNode().execute({"rows": 100, "cols": 4, "seed": 0})["df"]
    split = PdXYSplitNode().execute({"df": df, "label_col": "label"})
    r = SkTrainTestSplitNode().execute({"X": split["X"], "y": split["y"], "test_size": 0.2, "random_state": 42})
    assert len(r["X_train"]) == 80
    assert len(r["X_test"]) == 20

def test_sk_logistic_regression():
    from nodes.pandas import PdMakeSampleNode, PdXYSplitNode
    from nodes.sklearn import SkTrainTestSplitNode, SkLogisticRegressionNode, SkPredictNode, SkAccuracyNode
    df = PdMakeSampleNode().execute({"rows": 200, "cols": 4, "seed": 0})["df"]
    split = PdXYSplitNode().execute({"df": df, "label_col": "label"})
    tt = SkTrainTestSplitNode().execute({"X": split["X"], "y": split["y"], "test_size": 0.2, "random_state": 42})
    import pandas as pd
    X_train = tt["X_train"].values if hasattr(tt["X_train"], "values") else tt["X_train"]
    X_test  = tt["X_test"].values  if hasattr(tt["X_test"],  "values") else tt["X_test"]
    y_train = tt["y_train"].values if hasattr(tt["y_train"], "values") else tt["y_train"]
    y_test  = tt["y_test"].values  if hasattr(tt["y_test"],  "values") else tt["y_test"]
    model_r = SkLogisticRegressionNode().execute({"X_train": X_train, "y_train": y_train, "max_iter": 1000})
    preds = SkPredictNode().execute({"model": model_r["model"], "X": X_test})
    acc = SkAccuracyNode().execute({"y_true": y_test, "y_pred": preds["predictions"]})
    assert acc["accuracy"] > 0.4  # random data, just check it runs

def test_sk_pca():
    from nodes.sklearn import SkPCANode
    X = np.random.randn(50, 5).astype(np.float32)
    r = SkPCANode().execute({"X": X, "n_components": 2})
    assert r["transformed"].shape == (50, 2)

def test_sk_none_guard():
    from nodes.sklearn import SkPredictNode
    r = SkPredictNode().execute({"model": None, "X": None})
    assert r["predictions"] is None

# ── SciPy ──────────────────────────────────────────────────────────────────

def test_sp_ttest():
    from nodes.scipy import SpTTestNode
    a = np.random.randn(30)
    b = np.random.randn(30) + 1.0
    r = SpTTestNode().execute({"a": a, "b": b})
    assert "statistic" in r
    assert isinstance(r["pvalue"], float)

def test_sp_fft():
    from nodes.scipy import SpFFTNode
    signal = np.sin(np.linspace(0, 2*np.pi, 256))
    r = SpFFTNode().execute({"signal": signal})
    assert len(r["frequencies"]) > 0

def test_sp_histogram():
    from nodes.scipy import SpHistogramNode
    arr = np.random.randn(100)
    r = SpHistogramNode().execute({"array": arr, "bins": 10})
    assert len(r["counts"]) == 10

# ── Viz ────────────────────────────────────────────────────────────────────

def test_viz_line():
    from nodes.viz import VizLineNode
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    r = VizLineNode().execute({"x": x, "y": y, "title": "Test", "xlabel": "x", "ylabel": "y", "color": "cyan"})
    assert r["image"] is not None
    assert r["image"].ndim == 3
    assert r["image"].shape[2] == 3

def test_viz_hist():
    from nodes.viz import VizHistNode
    arr = np.random.randn(200)
    r = VizHistNode().execute({"array": arr, "bins": 20, "title": "Hist", "color": "steelblue"})
    assert r["image"] is not None

def test_viz_heatmap():
    from nodes.viz import VizHeatmapNode
    mat = np.random.rand(5, 5)
    r = VizHeatmapNode().execute({"matrix": mat, "title": "Heat", "cmap": "viridis"})
    assert r["image"] is not None
    assert r["image"].shape[2] == 3

def test_viz_none_guard():
    from nodes.viz import VizLineNode
    r = VizLineNode().execute({"x": None, "y": None, "title": "T", "xlabel": "x", "ylabel": "y", "color": "cyan"})
    assert r["image"] is None
