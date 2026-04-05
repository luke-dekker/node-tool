"""Scikit-learn model nodes."""
from core.node import BaseNode, PortType

C = "Sklearn"


class SkLinearRegressionNode(BaseNode):
    type_name = "sk_linear_regression"
    label = "Linear Regression"
    category = C
    description = "Fit LinearRegression on X_train, y_train"

    def _setup_ports(self):
        self.add_input("X_train", PortType.NDARRAY)
        self.add_input("y_train", PortType.NDARRAY)
        self.add_output("model",  PortType.SKLEARN_MODEL)

    def execute(self, inputs):
        try:
            from sklearn.linear_model import LinearRegression
            X, y = inputs.get("X_train"), inputs.get("y_train")
            if X is None or y is None:
                return {"model": None}
            m = LinearRegression(); m.fit(X, y)
            return {"model": m}
        except Exception:
            return {"model": None}


class SkLogisticRegressionNode(BaseNode):
    type_name = "sk_logistic_regression"
    label = "Logistic Regression"
    category = C
    description = "Fit LogisticRegression on X_train, y_train"

    def _setup_ports(self):
        self.add_input("X_train",  PortType.NDARRAY)
        self.add_input("y_train",  PortType.NDARRAY)
        self.add_input("max_iter", PortType.INT, 1000)
        self.add_output("model",   PortType.SKLEARN_MODEL)

    def execute(self, inputs):
        try:
            from sklearn.linear_model import LogisticRegression
            X, y = inputs.get("X_train"), inputs.get("y_train")
            if X is None or y is None:
                return {"model": None}
            m = LogisticRegression(max_iter=int(inputs.get("max_iter", 1000)))
            m.fit(X, y); return {"model": m}
        except Exception:
            return {"model": None}


class SkRandomForestNode(BaseNode):
    type_name = "sk_random_forest"
    label = "Random Forest"
    category = C
    description = "Fit RandomForestClassifier on X_train, y_train"

    def _setup_ports(self):
        self.add_input("X_train",      PortType.NDARRAY)
        self.add_input("y_train",      PortType.NDARRAY)
        self.add_input("n_estimators", PortType.INT, 100)
        self.add_input("random_state", PortType.INT, 42)
        self.add_output("model",       PortType.SKLEARN_MODEL)

    def execute(self, inputs):
        try:
            from sklearn.ensemble import RandomForestClassifier
            X, y = inputs.get("X_train"), inputs.get("y_train")
            if X is None or y is None:
                return {"model": None}
            m = RandomForestClassifier(n_estimators=int(inputs.get("n_estimators", 100)),
                                       random_state=int(inputs.get("random_state", 42)))
            m.fit(X, y); return {"model": m}
        except Exception:
            return {"model": None}


class SkSVCNode(BaseNode):
    type_name = "sk_svc"
    label = "SVC"
    category = C
    description = "Fit SVC on X_train, y_train"

    def _setup_ports(self):
        self.add_input("X_train", PortType.NDARRAY)
        self.add_input("y_train", PortType.NDARRAY)
        self.add_input("C",       PortType.FLOAT,  1.0)
        self.add_input("kernel",  PortType.STRING, "rbf")
        self.add_output("model",  PortType.SKLEARN_MODEL)

    def execute(self, inputs):
        try:
            from sklearn.svm import SVC
            X, y = inputs.get("X_train"), inputs.get("y_train")
            if X is None or y is None:
                return {"model": None}
            m = SVC(C=float(inputs.get("C", 1.0)), kernel=inputs.get("kernel") or "rbf")
            m.fit(X, y); return {"model": m}
        except Exception:
            return {"model": None}


class SkGradientBoostingNode(BaseNode):
    type_name = "sk_gradient_boosting"
    label = "Gradient Boosting"
    category = C
    description = "GradientBoostingClassifier"

    def _setup_ports(self):
        self.add_input("X_train",       PortType.NDARRAY)
        self.add_input("y_train",       PortType.NDARRAY)
        self.add_input("n_estimators",  PortType.INT,   100)
        self.add_input("learning_rate", PortType.FLOAT, 0.1)
        self.add_output("model",        PortType.SKLEARN_MODEL)

    def execute(self, inputs):
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            X, y = inputs.get("X_train"), inputs.get("y_train")
            if X is None or y is None:
                return {"model": None}
            m = GradientBoostingClassifier(n_estimators=int(inputs.get("n_estimators", 100)),
                                           learning_rate=float(inputs.get("learning_rate", 0.1)))
            m.fit(X, y); return {"model": m}
        except Exception:
            return {"model": None}


class SkKMeansNode(BaseNode):
    type_name = "sk_kmeans"
    label = "KMeans"
    category = C
    description = "KMeans clustering — fit_predict"

    def _setup_ports(self):
        self.add_input("X",            PortType.NDARRAY)
        self.add_input("n_clusters",   PortType.INT, 3)
        self.add_input("random_state", PortType.INT, 42)
        self.add_output("model",       PortType.SKLEARN_MODEL)
        self.add_output("labels",      PortType.NDARRAY)

    def execute(self, inputs):
        null = {"model": None, "labels": None}
        try:
            from sklearn.cluster import KMeans
            X = inputs.get("X")
            if X is None:
                return null
            m = KMeans(n_clusters=int(inputs.get("n_clusters", 3)),
                       random_state=int(inputs.get("random_state", 42)), n_init="auto")
            labels = m.fit_predict(X)
            return {"model": m, "labels": labels}
        except Exception:
            return null


class SkPCANode(BaseNode):
    type_name = "sk_pca"
    label = "PCA"
    category = C
    description = "PCA dimensionality reduction — fit_transform"

    def _setup_ports(self):
        self.add_input("X",            PortType.NDARRAY)
        self.add_input("n_components", PortType.INT, 2)
        self.add_output("transformed", PortType.NDARRAY)
        self.add_output("model",       PortType.SKLEARN_MODEL)

    def execute(self, inputs):
        null = {"transformed": None, "model": None}
        try:
            from sklearn.decomposition import PCA
            X = inputs.get("X")
            if X is None:
                return null
            m = PCA(n_components=int(inputs.get("n_components", 2)))
            return {"transformed": m.fit_transform(X), "model": m}
        except Exception:
            return null


class SkPredictNode(BaseNode):
    type_name = "sk_predict"
    label = "Predict"
    category = C
    description = "model.predict(X)"

    def _setup_ports(self):
        self.add_input("model",        PortType.SKLEARN_MODEL)
        self.add_input("X",            PortType.NDARRAY)
        self.add_output("predictions", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            model, X = inputs.get("model"), inputs.get("X")
            if model is None or X is None:
                return {"predictions": None}
            return {"predictions": model.predict(X)}
        except Exception:
            return {"predictions": None}


class SkPredictProbaNode(BaseNode):
    type_name = "sk_predict_proba"
    label = "Predict Proba"
    category = C
    description = "model.predict_proba(X)"

    def _setup_ports(self):
        self.add_input("model",          PortType.SKLEARN_MODEL)
        self.add_input("X",              PortType.NDARRAY)
        self.add_output("probabilities", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            model, X = inputs.get("model"), inputs.get("X")
            if model is None or X is None:
                return {"probabilities": None}
            return {"probabilities": model.predict_proba(X)}
        except Exception:
            return {"probabilities": None}
