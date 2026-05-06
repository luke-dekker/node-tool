"""Tests for numpy, pandas, sklearn, scipy, viz nodes — headless, no DPG."""
import pytest
import numpy as np

# ── NumPy ──────────────────────────────────────────────────────────────────

def test_np_arange():
    from nodes.numpy import NpCreateNode
    r = NpCreateNode().execute({"kind": "arange", "start": 0, "stop": 5, "step": 1})
    assert list(r["array"]) == [0,1,2,3,4]

def test_np_linspace():
    from nodes.numpy import NpCreateNode
    r = NpCreateNode().execute({"kind": "linspace", "start": 0, "stop": 1, "num": 5})
    assert len(r["array"]) == 5
    assert abs(r["array"][-1] - 1.0) < 1e-6

def test_np_zeros():
    from nodes.numpy import NpCreateNode
    r = NpCreateNode().execute({"kind": "zeros", "shape": "2,3"})
    assert r["array"].shape == (2, 3)

def test_np_reshape():
    from nodes.numpy import NpCreateNode, NpOpNode
    arr = NpCreateNode().execute({"kind": "arange", "start": 0, "stop": 6, "step": 1})["array"]
    r = NpOpNode().execute({"array": arr, "op": "reshape", "shape": "2,3"})
    assert r["result"].shape == (2, 3)

def test_np_dot():
    from nodes.numpy import NpLinalgNode
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    r = NpLinalgNode().execute({"kind": "dot", "a": a, "b": b})
    assert abs(r["result"] - 32.0) < 1e-6

def test_np_none_guard():
    from nodes.numpy import NpOpNode
    r = NpOpNode().execute({"array": None, "op": "mean", "axis": -99})
    assert r["result"] is None

# ── Pandas ─────────────────────────────────────────────────────────────────

def test_pd_make_sample():
    from nodes.pandas import PdMakeSampleNode
    r = PdMakeSampleNode().execute({"kind": "sample", "rows": 50, "cols": 3, "seed": 0})
    import pandas as pd
    assert isinstance(r["df"], pd.DataFrame)
    assert r["df"].shape == (50, 4)  # 3 cols + label

def test_pd_filter():
    from nodes.pandas import PdMakeSampleNode, PdTransformNode
    df = PdMakeSampleNode().execute({"kind": "sample", "rows": 100, "cols": 2, "seed": 42})["df"]
    r = PdTransformNode().execute({
        "df": df, "op": "filter_rows", "column": "label", "compare": "==", "value": 1.0,
    })
    assert (r["result"]["label"] == 1).all()

def test_pd_to_numpy():
    from nodes.pandas import PdMakeSampleNode, PdToNumpyNode
    df = PdMakeSampleNode().execute({"kind": "sample", "rows": 10, "cols": 2, "seed": 0})["df"]
    r = PdToNumpyNode().execute({"df": df, "op": "to_numpy"})
    assert r["array"].shape[0] == 10

def test_pd_xy_split():
    from nodes.pandas import PdMakeSampleNode, PdXYSplitNode
    df = PdMakeSampleNode().execute({"kind": "sample", "rows": 20, "cols": 3, "seed": 0})["df"]
    r = PdXYSplitNode().execute({"df": df, "op": "xy_split", "label_col": "label"})
    # PdXYSplitNode aliases to PdTransformNode: X→`result`, y→`series`.
    assert "label" not in r["result"].columns
    assert r["series"].name == "label"

# ── Sklearn ────────────────────────────────────────────────────────────────

def test_sk_train_test_split():
    from nodes.pandas import PdMakeSampleNode, PdXYSplitNode
    from nodes.sklearn import SkTrainTestSplitNode
    df = PdMakeSampleNode().execute({"kind": "sample", "rows": 100, "cols": 4, "seed": 0})["df"]
    split = PdXYSplitNode().execute({"df": df, "op": "xy_split", "label_col": "label"})
    r = SkTrainTestSplitNode().execute({"X": split["result"], "y": split["series"], "test_size": 0.2, "random_state": 42})
    assert len(r["X_train"]) == 80
    assert len(r["X_test"]) == 20

def test_sk_classifier():
    from nodes.pandas import PdMakeSampleNode, PdXYSplitNode
    from nodes.sklearn import SkTrainTestSplitNode, SkClassifierNode, SkPredictNode, SkAccuracyNode
    df = PdMakeSampleNode().execute({"kind": "sample", "rows": 200, "cols": 4, "seed": 0})["df"]
    split = PdXYSplitNode().execute({"df": df, "op": "xy_split", "label_col": "label"})
    tt = SkTrainTestSplitNode().execute({"X": split["result"], "y": split["series"], "test_size": 0.2, "random_state": 42})
    import pandas as pd
    X_train = tt["X_train"].values if hasattr(tt["X_train"], "values") else tt["X_train"]
    X_test  = tt["X_test"].values  if hasattr(tt["X_test"],  "values") else tt["X_test"]
    y_train = tt["y_train"].values if hasattr(tt["y_train"], "values") else tt["y_train"]
    y_test  = tt["y_test"].values  if hasattr(tt["y_test"],  "values") else tt["y_test"]
    model_r = SkClassifierNode().execute({"algorithm": "logistic_regression", "X_train": X_train, "y_train": y_train, "max_iter": 1000})
    preds = SkPredictNode().execute({"model": model_r["model"], "X": X_test})
    # SkAccuracyNode is now an alias for SkMetricsNode; output key is "value"
    acc = SkAccuracyNode().execute({"y_true": y_test, "y_pred": preds["predictions"], "metric": "accuracy"})
    assert acc["value"] > 0.4  # random data, just check it runs

def test_sk_pca():
    from nodes.sklearn import SkPCANode
    X = np.random.randn(50, 5).astype(np.float32)
    # SkPCANode aliases SkModelNode; pass algorithm="pca" + X via X_train.
    r = SkPCANode().execute({"algorithm": "pca", "X_train": X, "n_components": 2})
    assert r["transformed"].shape == (50, 2)

def test_sk_none_guard():
    from nodes.sklearn import SkPredictNode
    r = SkPredictNode().execute({"model": None, "X": None})
    assert r["predictions"] is None

# ── SciPy ──────────────────────────────────────────────────────────────────

def test_sp_ttest():
    from nodes.scipy import SpStatsNode
    a = np.random.randn(30)
    b = np.random.randn(30) + 1.0
    r = SpStatsNode().execute({"op": "ttest", "x": a, "y": b})
    assert "statistic" in r
    assert isinstance(r["pvalue"], float)

def test_sp_fft():
    from nodes.scipy import SpSignalNode
    signal = np.sin(np.linspace(0, 2*np.pi, 256))
    r = SpSignalNode().execute({"op": "fft", "a": signal})
    # fft: primary=magnitude, secondary=frequencies
    assert r["secondary"] is not None and len(r["secondary"]) > 0

def test_sp_histogram():
    from nodes.scipy import SpSignalNode
    arr = np.random.randn(100)
    r = SpSignalNode().execute({"op": "histogram", "a": arr, "bins": 10})
    # histogram: primary=counts, secondary=bin_edges
    assert len(r["primary"]) == 10

# ── Viz ────────────────────────────────────────────────────────────────────

def test_viz_line():
    from nodes.viz import VizPlotNode
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    r = VizPlotNode().execute({"kind": "line", "data": x, "data2": y,
                               "title": "Test", "xlabel": "x", "ylabel": "y",
                               "color": "cyan"})
    assert r["image"] is not None
    assert r["image"].ndim == 3
    assert r["image"].shape[2] == 3

def test_viz_hist():
    from nodes.viz import VizPlotNode
    arr = np.random.randn(200)
    r = VizPlotNode().execute({"kind": "hist", "data": arr, "bins": 20,
                               "title": "Hist", "color": "steelblue"})
    assert r["image"] is not None

def test_viz_heatmap():
    from nodes.viz import VizPlotNode
    mat = np.random.rand(5, 5)
    r = VizPlotNode().execute({"kind": "heatmap", "data": mat,
                               "title": "Heat", "cmap": "viridis"})
    assert r["image"] is not None
    assert r["image"].shape[2] == 3

def test_viz_none_guard():
    from nodes.viz import VizPlotNode
    r = VizPlotNode().execute({"kind": "line", "data": None, "data2": None,
                               "title": "T"})
    assert r["image"] is None
