"""Consolidated sklearn model node.

Replaces SkClassifierNode (already a 4-algo factory), SkLinearRegressionNode,
SkKMeansNode, SkPCANode. Pick `algorithm`:
  classification: logistic_regression | random_forest | svc | gradient_boosting
  regression:     linear_regression
  clustering:     kmeans
  dim. reduce:    pca

Outputs (each populated by relevant algorithms):
  model       — fitted estimator (always)
  labels      — np.ndarray (kmeans only — cluster assignments)
  transformed — np.ndarray (pca only — projected data)
"""
from core.node import BaseNode, PortType


_ALGOS = ["logistic_regression", "random_forest", "svc", "gradient_boosting",
          "linear_regression", "kmeans", "pca"]


class SkModelNode(BaseNode):
    type_name   = "sk_model"
    label       = "Model"
    category    = "Sklearn"
    description = (
        "Fit a sklearn model. Pick `algorithm` from the dropdown — the "
        "inspector hides hyperparameters that don't apply.\n"
        "  classification: logistic_regression / random_forest / svc / gradient_boosting\n"
        "  regression:     linear_regression\n"
        "  clustering:     kmeans  (also outputs `labels`)\n"
        "  dim. reduce:    pca     (also outputs `transformed`)"
    )

    def relevant_inputs(self, values):
        algo = (values.get("algorithm") or "logistic_regression").strip()
        per_algo = {
            "logistic_regression": ["max_iter"],
            "random_forest":       ["n_estimators", "random_state"],
            "svc":                 ["C", "kernel"],
            "gradient_boosting":   ["n_estimators", "learning_rate"],
            "linear_regression":   [],
            "kmeans":              ["n_clusters", "random_state"],
            "pca":                 ["n_components"],
        }
        return ["algorithm"] + per_algo.get(algo, [])

    def _setup_ports(self):
        self.add_input("algorithm",     PortType.STRING, "logistic_regression", choices=_ALGOS)
        # supervised inputs (X_train + y_train); unsupervised reads X_train as X
        self.add_input("X_train",       PortType.NDARRAY)
        self.add_input("y_train",       PortType.NDARRAY, optional=True)
        # union of hyperparams
        self.add_input("max_iter",      PortType.INT,   1000, optional=True)
        self.add_input("n_estimators",  PortType.INT,   100,  optional=True)
        self.add_input("learning_rate", PortType.FLOAT, 0.1,  optional=True)
        self.add_input("C",             PortType.FLOAT, 1.0,  optional=True)
        self.add_input("kernel",        PortType.STRING, "rbf", optional=True)
        self.add_input("random_state",  PortType.INT,   42,   optional=True)
        self.add_input("n_clusters",    PortType.INT,   3,    optional=True)
        self.add_input("n_components",  PortType.INT,   2,    optional=True)
        self.add_output("model",       PortType.SKLEARN_MODEL)
        self.add_output("labels",      PortType.NDARRAY)
        self.add_output("transformed", PortType.NDARRAY)

    def execute(self, inputs):
        out = {"model": None, "labels": None, "transformed": None}
        try:
            algo = (inputs.get("algorithm") or "logistic_regression").strip()
            X = inputs.get("X_train")
            y = inputs.get("y_train")
            if X is None:
                return out

            if algo == "logistic_regression":
                from sklearn.linear_model import LogisticRegression
                m = LogisticRegression(max_iter=int(inputs.get("max_iter", 1000)))
                if y is None: return out
                m.fit(X, y); out["model"] = m; return out
            if algo == "random_forest":
                from sklearn.ensemble import RandomForestClassifier
                m = RandomForestClassifier(
                    n_estimators=int(inputs.get("n_estimators", 100)),
                    random_state=int(inputs.get("random_state", 42)))
                if y is None: return out
                m.fit(X, y); out["model"] = m; return out
            if algo == "svc":
                from sklearn.svm import SVC
                m = SVC(C=float(inputs.get("C", 1.0)),
                        kernel=inputs.get("kernel") or "rbf")
                if y is None: return out
                m.fit(X, y); out["model"] = m; return out
            if algo == "gradient_boosting":
                from sklearn.ensemble import GradientBoostingClassifier
                m = GradientBoostingClassifier(
                    n_estimators=int(inputs.get("n_estimators", 100)),
                    learning_rate=float(inputs.get("learning_rate", 0.1)))
                if y is None: return out
                m.fit(X, y); out["model"] = m; return out
            if algo == "linear_regression":
                from sklearn.linear_model import LinearRegression
                m = LinearRegression()
                if y is None: return out
                m.fit(X, y); out["model"] = m; return out
            if algo == "kmeans":
                from sklearn.cluster import KMeans
                m = KMeans(n_clusters=int(inputs.get("n_clusters", 3)),
                           random_state=int(inputs.get("random_state", 42)),
                           n_init="auto")
                out["labels"] = m.fit_predict(X)
                out["model"]  = m
                return out
            if algo == "pca":
                from sklearn.decomposition import PCA
                m = PCA(n_components=int(inputs.get("n_components", 2)))
                out["transformed"] = m.fit_transform(X)
                out["model"]       = m
                return out
            return out
        except Exception:
            return {"model": None, "labels": None, "transformed": None}

    def export(self, iv, ov):
        algo = (self.inputs["algorithm"].default_value or "logistic_regression")
        X = self._val(iv, "X_train"); y = self._val(iv, "y_train")
        m = ov.get("model", "_sk_model")

        if algo == "logistic_regression":
            mi = self._val(iv, "max_iter")
            return ["from sklearn.linear_model import LogisticRegression"], [
                f"{m} = LogisticRegression(max_iter={mi})", f"{m}.fit({X}, {y})",
            ]
        if algo == "random_forest":
            n  = self._val(iv, "n_estimators"); rs = self._val(iv, "random_state")
            return ["from sklearn.ensemble import RandomForestClassifier"], [
                f"{m} = RandomForestClassifier(n_estimators={n}, random_state={rs})",
                f"{m}.fit({X}, {y})",
            ]
        if algo == "svc":
            C = self._val(iv, "C"); k = self._val(iv, "kernel")
            return ["from sklearn.svm import SVC"], [
                f"{m} = SVC(C={C}, kernel={k})", f"{m}.fit({X}, {y})",
            ]
        if algo == "gradient_boosting":
            n  = self._val(iv, "n_estimators"); lr = self._val(iv, "learning_rate")
            return ["from sklearn.ensemble import GradientBoostingClassifier"], [
                f"{m} = GradientBoostingClassifier(n_estimators={n}, learning_rate={lr})",
                f"{m}.fit({X}, {y})",
            ]
        if algo == "linear_regression":
            return ["from sklearn.linear_model import LinearRegression"], [
                f"{m} = LinearRegression()", f"{m}.fit({X}, {y})",
            ]
        if algo == "kmeans":
            k  = self._val(iv, "n_clusters"); rs = self._val(iv, "random_state")
            labels = ov.get("labels", "_sk_labels")
            return ["from sklearn.cluster import KMeans"], [
                f"{m} = KMeans(n_clusters={k}, random_state={rs}, n_init='auto')",
                f"{labels} = {m}.fit_predict({X})",
            ]
        if algo == "pca":
            n = self._val(iv, "n_components"); tr = ov.get("transformed", "_sk_transformed")
            return ["from sklearn.decomposition import PCA"], [
                f"{m} = PCA(n_components={n})", f"{tr} = {m}.fit_transform({X})",
            ]
        return [], [f"# unknown sk algorithm {algo!r}"]
