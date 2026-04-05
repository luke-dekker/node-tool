"""Scikit-learn nodes — preprocessing, models, metrics."""
from __future__ import annotations
from core.node import BaseNode
from core.node import PortType

CATEGORY = "Sklearn"


# ── Preprocessing ─────────────────────────────────────────────────────────────

class SkTrainTestSplitNode(BaseNode):
    type_name = "sk_train_test_split"
    label = "Train/Test Split"
    category = CATEGORY
    description = "sklearn train_test_split on X (DataFrame) and y (Series)"

    def _setup_ports(self):
        self.add_input("X",            PortType.DATAFRAME)
        self.add_input("y",            PortType.SERIES)
        self.add_input("test_size",    PortType.FLOAT, 0.2)
        self.add_input("random_state", PortType.INT,   42)
        self.add_output("X_train",     PortType.DATAFRAME)
        self.add_output("X_test",      PortType.DATAFRAME)
        self.add_output("y_train",     PortType.SERIES)
        self.add_output("y_test",      PortType.SERIES)

    def execute(self, inputs):
        null = {"X_train": None, "X_test": None, "y_train": None, "y_test": None}
        try:
            from sklearn.model_selection import train_test_split
            X  = inputs.get("X")
            y  = inputs.get("y")
            ts = inputs.get("test_size",    0.2)
            rs = inputs.get("random_state", 42)
            if X is None or y is None:
                return null
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=float(ts), random_state=int(rs)
            )
            return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}
        except Exception:
            return null


class SkStandardScalerNode(BaseNode):
    type_name = "sk_standard_scaler"
    label = "Standard Scaler"
    category = CATEGORY
    description = "Fit StandardScaler on X_train, transform both splits"

    def _setup_ports(self):
        self.add_input("X_train",        PortType.DATAFRAME)
        self.add_input("X_test",         PortType.DATAFRAME)
        self.add_output("X_train_scaled", PortType.NDARRAY)
        self.add_output("X_test_scaled",  PortType.NDARRAY)
        self.add_output("scaler",         PortType.SKLEARN_MODEL)

    def execute(self, inputs):
        null = {"X_train_scaled": None, "X_test_scaled": None, "scaler": None}
        try:
            from sklearn.preprocessing import StandardScaler
            X_train = inputs.get("X_train")
            X_test  = inputs.get("X_test")
            if X_train is None or X_test is None:
                return null
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled  = scaler.transform(X_test)
            return {"X_train_scaled": X_train_scaled, "X_test_scaled": X_test_scaled, "scaler": scaler}
        except Exception:
            return null


class SkMinMaxScalerNode(BaseNode):
    type_name = "sk_minmax_scaler"
    label = "MinMax Scaler"
    category = CATEGORY
    description = "Fit MinMaxScaler on X_train, transform both splits"

    def _setup_ports(self):
        self.add_input("X_train",        PortType.DATAFRAME)
        self.add_input("X_test",         PortType.DATAFRAME)
        self.add_output("X_train_scaled", PortType.NDARRAY)
        self.add_output("X_test_scaled",  PortType.NDARRAY)
        self.add_output("scaler",         PortType.SKLEARN_MODEL)

    def execute(self, inputs):
        null = {"X_train_scaled": None, "X_test_scaled": None, "scaler": None}
        try:
            from sklearn.preprocessing import MinMaxScaler
            X_train = inputs.get("X_train")
            X_test  = inputs.get("X_test")
            if X_train is None or X_test is None:
                return null
            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled  = scaler.transform(X_test)
            return {"X_train_scaled": X_train_scaled, "X_test_scaled": X_test_scaled, "scaler": scaler}
        except Exception:
            return null


class SkLabelEncoderNode(BaseNode):
    type_name = "sk_label_encoder"
    label = "Label Encoder"
    category = CATEGORY
    description = "Fit LabelEncoder on series, return encoded ndarray + encoder"

    def _setup_ports(self):
        self.add_input("series",  PortType.SERIES)
        self.add_output("encoded", PortType.NDARRAY)
        self.add_output("encoder", PortType.SKLEARN_MODEL)

    def execute(self, inputs):
        null = {"encoded": None, "encoder": None}
        try:
            from sklearn.preprocessing import LabelEncoder
            series = inputs.get("series")
            if series is None:
                return null
            enc = LabelEncoder()
            encoded = enc.fit_transform(series)
            return {"encoded": encoded, "encoder": enc}
        except Exception:
            return null


class SkOneHotEncoderNode(BaseNode):
    type_name = "sk_onehot_encoder"
    label = "One-Hot Encoder"
    category = CATEGORY
    description = "OneHotEncoder on ndarray (sparse_output=False)"

    def _setup_ports(self):
        self.add_input("array",   PortType.NDARRAY)
        self.add_output("encoded", PortType.NDARRAY)
        self.add_output("encoder", PortType.SKLEARN_MODEL)

    def execute(self, inputs):
        null = {"encoded": None, "encoder": None}
        try:
            import numpy as np
            from sklearn.preprocessing import OneHotEncoder
            array = inputs.get("array")
            if array is None:
                return null
            if array.ndim == 1:
                array = array.reshape(-1, 1)
            enc = OneHotEncoder(sparse_output=False)
            encoded = enc.fit_transform(array)
            return {"encoded": encoded, "encoder": enc}
        except Exception:
            return null


# ── Models ────────────────────────────────────────────────────────────────────

class SkLinearRegressionNode(BaseNode):
    type_name = "sk_linear_regression"
    label = "Linear Regression"
    category = CATEGORY
    description = "Fit LinearRegression on X_train, y_train"

    def _setup_ports(self):
        self.add_input("X_train", PortType.NDARRAY)
        self.add_input("y_train", PortType.NDARRAY)
        self.add_output("model",  PortType.SKLEARN_MODEL)

    def execute(self, inputs):
        null = {"model": None}
        try:
            from sklearn.linear_model import LinearRegression
            X = inputs.get("X_train")
            y = inputs.get("y_train")
            if X is None or y is None:
                return null
            model = LinearRegression()
            model.fit(X, y)
            return {"model": model}
        except Exception:
            return null


class SkLogisticRegressionNode(BaseNode):
    type_name = "sk_logistic_regression"
    label = "Logistic Regression"
    category = CATEGORY
    description = "Fit LogisticRegression on X_train, y_train"

    def _setup_ports(self):
        self.add_input("X_train",  PortType.NDARRAY)
        self.add_input("y_train",  PortType.NDARRAY)
        self.add_input("max_iter", PortType.INT, 1000)
        self.add_output("model",   PortType.SKLEARN_MODEL)

    def execute(self, inputs):
        null = {"model": None}
        try:
            from sklearn.linear_model import LogisticRegression
            X        = inputs.get("X_train")
            y        = inputs.get("y_train")
            max_iter = inputs.get("max_iter", 1000)
            if X is None or y is None:
                return null
            model = LogisticRegression(max_iter=int(max_iter))
            model.fit(X, y)
            return {"model": model}
        except Exception:
            return null


class SkRandomForestNode(BaseNode):
    type_name = "sk_random_forest"
    label = "Random Forest"
    category = CATEGORY
    description = "Fit RandomForestClassifier on X_train, y_train"

    def _setup_ports(self):
        self.add_input("X_train",      PortType.NDARRAY)
        self.add_input("y_train",      PortType.NDARRAY)
        self.add_input("n_estimators", PortType.INT, 100)
        self.add_input("random_state", PortType.INT, 42)
        self.add_output("model",       PortType.SKLEARN_MODEL)

    def execute(self, inputs):
        null = {"model": None}
        try:
            from sklearn.ensemble import RandomForestClassifier
            X  = inputs.get("X_train")
            y  = inputs.get("y_train")
            n  = inputs.get("n_estimators", 100)
            rs = inputs.get("random_state", 42)
            if X is None or y is None:
                return null
            model = RandomForestClassifier(n_estimators=int(n), random_state=int(rs))
            model.fit(X, y)
            return {"model": model}
        except Exception:
            return null


class SkSVCNode(BaseNode):
    type_name = "sk_svc"
    label = "SVC"
    category = CATEGORY
    description = "Fit SVC on X_train, y_train"

    def _setup_ports(self):
        self.add_input("X_train", PortType.NDARRAY)
        self.add_input("y_train", PortType.NDARRAY)
        self.add_input("C",       PortType.FLOAT,  1.0)
        self.add_input("kernel",  PortType.STRING, "rbf")
        self.add_output("model",  PortType.SKLEARN_MODEL)

    def execute(self, inputs):
        null = {"model": None}
        try:
            from sklearn.svm import SVC
            X      = inputs.get("X_train")
            y      = inputs.get("y_train")
            C      = inputs.get("C", 1.0)
            kernel = inputs.get("kernel", "rbf") or "rbf"
            if X is None or y is None:
                return null
            model = SVC(C=float(C), kernel=kernel)
            model.fit(X, y)
            return {"model": model}
        except Exception:
            return null


class SkKMeansNode(BaseNode):
    type_name = "sk_kmeans"
    label = "KMeans"
    category = CATEGORY
    description = "KMeans clustering — fit_predict"

    def _setup_ports(self):
        self.add_input("X",           PortType.NDARRAY)
        self.add_input("n_clusters",  PortType.INT, 3)
        self.add_input("random_state", PortType.INT, 42)
        self.add_output("model",      PortType.SKLEARN_MODEL)
        self.add_output("labels",     PortType.NDARRAY)

    def execute(self, inputs):
        null = {"model": None, "labels": None}
        try:
            from sklearn.cluster import KMeans
            X  = inputs.get("X")
            k  = inputs.get("n_clusters",  3)
            rs = inputs.get("random_state", 42)
            if X is None:
                return null
            model  = KMeans(n_clusters=int(k), random_state=int(rs), n_init="auto")
            labels = model.fit_predict(X)
            return {"model": model, "labels": labels}
        except Exception:
            return null


class SkPCANode(BaseNode):
    type_name = "sk_pca"
    label = "PCA"
    category = CATEGORY
    description = "PCA dimensionality reduction — fit_transform"

    def _setup_ports(self):
        self.add_input("X",             PortType.NDARRAY)
        self.add_input("n_components",  PortType.INT, 2)
        self.add_output("transformed",  PortType.NDARRAY)
        self.add_output("model",        PortType.SKLEARN_MODEL)

    def execute(self, inputs):
        null = {"transformed": None, "model": None}
        try:
            from sklearn.decomposition import PCA
            X = inputs.get("X")
            n = inputs.get("n_components", 2)
            if X is None:
                return null
            model = PCA(n_components=int(n))
            transformed = model.fit_transform(X)
            return {"transformed": transformed, "model": model}
        except Exception:
            return null


class SkGradientBoostingNode(BaseNode):
    type_name = "sk_gradient_boosting"
    label = "Gradient Boosting"
    category = CATEGORY
    description = "GradientBoostingClassifier"

    def _setup_ports(self):
        self.add_input("X_train",       PortType.NDARRAY)
        self.add_input("y_train",       PortType.NDARRAY)
        self.add_input("n_estimators",  PortType.INT,   100)
        self.add_input("learning_rate", PortType.FLOAT, 0.1)
        self.add_output("model",        PortType.SKLEARN_MODEL)

    def execute(self, inputs):
        null = {"model": None}
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            X  = inputs.get("X_train")
            y  = inputs.get("y_train")
            n  = inputs.get("n_estimators",  100)
            lr = inputs.get("learning_rate", 0.1)
            if X is None or y is None:
                return null
            model = GradientBoostingClassifier(n_estimators=int(n), learning_rate=float(lr))
            model.fit(X, y)
            return {"model": model}
        except Exception:
            return null


class SkPredictNode(BaseNode):
    type_name = "sk_predict"
    label = "Predict"
    category = CATEGORY
    description = "model.predict(X)"

    def _setup_ports(self):
        self.add_input("model",       PortType.SKLEARN_MODEL)
        self.add_input("X",           PortType.NDARRAY)
        self.add_output("predictions", PortType.NDARRAY)

    def execute(self, inputs):
        null = {"predictions": None}
        try:
            model = inputs.get("model")
            X     = inputs.get("X")
            if model is None or X is None:
                return null
            return {"predictions": model.predict(X)}
        except Exception:
            return null


class SkPredictProbaNode(BaseNode):
    type_name = "sk_predict_proba"
    label = "Predict Proba"
    category = CATEGORY
    description = "model.predict_proba(X)"

    def _setup_ports(self):
        self.add_input("model",         PortType.SKLEARN_MODEL)
        self.add_input("X",             PortType.NDARRAY)
        self.add_output("probabilities", PortType.NDARRAY)

    def execute(self, inputs):
        null = {"probabilities": None}
        try:
            model = inputs.get("model")
            X     = inputs.get("X")
            if model is None or X is None:
                return null
            return {"probabilities": model.predict_proba(X)}
        except Exception:
            return null


# ── Metrics ───────────────────────────────────────────────────────────────────

class SkAccuracyNode(BaseNode):
    type_name = "sk_accuracy"
    label = "Accuracy"
    category = CATEGORY
    description = "accuracy_score + classification_report"

    def _setup_ports(self):
        self.add_input("y_true",  PortType.NDARRAY)
        self.add_input("y_pred",  PortType.NDARRAY)
        self.add_output("accuracy", PortType.FLOAT)
        self.add_output("report",   PortType.STRING)

    def execute(self, inputs):
        null = {"accuracy": None, "report": ""}
        try:
            from sklearn.metrics import accuracy_score, classification_report
            y_true = inputs.get("y_true")
            y_pred = inputs.get("y_pred")
            if y_true is None or y_pred is None:
                return null
            acc    = float(accuracy_score(y_true, y_pred))
            report = classification_report(y_true, y_pred, zero_division=0)
            return {"accuracy": acc, "report": report}
        except Exception:
            return null


class SkConfusionMatrixNode(BaseNode):
    type_name = "sk_confusion_matrix"
    label = "Confusion Matrix"
    category = CATEGORY
    description = "confusion_matrix(y_true, y_pred)"

    def _setup_ports(self):
        self.add_input("y_true",  PortType.NDARRAY)
        self.add_input("y_pred",  PortType.NDARRAY)
        self.add_output("matrix", PortType.NDARRAY)
        self.add_output("info",   PortType.STRING)

    def execute(self, inputs):
        null = {"matrix": None, "info": ""}
        try:
            from sklearn.metrics import confusion_matrix
            y_true = inputs.get("y_true")
            y_pred = inputs.get("y_pred")
            if y_true is None or y_pred is None:
                return null
            mat  = confusion_matrix(y_true, y_pred)
            info = f"Confusion matrix shape: {mat.shape}"
            return {"matrix": mat, "info": info}
        except Exception:
            return null


class SkR2ScoreNode(BaseNode):
    type_name = "sk_r2_score"
    label = "R2 Score"
    category = CATEGORY
    description = "r2_score and mean_squared_error"

    def _setup_ports(self):
        self.add_input("y_true", PortType.NDARRAY)
        self.add_input("y_pred", PortType.NDARRAY)
        self.add_output("r2",    PortType.FLOAT)
        self.add_output("mse",   PortType.FLOAT)

    def execute(self, inputs):
        null = {"r2": None, "mse": None}
        try:
            from sklearn.metrics import r2_score, mean_squared_error
            y_true = inputs.get("y_true")
            y_pred = inputs.get("y_pred")
            if y_true is None or y_pred is None:
                return null
            r2  = float(r2_score(y_true, y_pred))
            mse = float(mean_squared_error(y_true, y_pred))
            return {"r2": r2, "mse": mse}
        except Exception:
            return null


class SkCrossValScoreNode(BaseNode):
    type_name = "sk_cross_val_score"
    label = "Cross-Val Score"
    category = CATEGORY
    description = "cross_val_score(model, X, y, cv=cv)"

    def _setup_ports(self):
        self.add_input("model",  PortType.SKLEARN_MODEL)
        self.add_input("X",      PortType.NDARRAY)
        self.add_input("y",      PortType.NDARRAY)
        self.add_input("cv",     PortType.INT, 5)
        self.add_output("scores", PortType.NDARRAY)
        self.add_output("mean",   PortType.FLOAT)

    def execute(self, inputs):
        null = {"scores": None, "mean": None}
        try:
            from sklearn.model_selection import cross_val_score
            model = inputs.get("model")
            X     = inputs.get("X")
            y     = inputs.get("y")
            cv    = inputs.get("cv", 5)
            if model is None or X is None or y is None:
                return null
            scores = cross_val_score(model, X, y, cv=int(cv))
            return {"scores": scores, "mean": float(scores.mean())}
        except Exception:
            return null
