"""Consolidated sklearn classifier node — replaces sk_logistic_regression,
sk_random_forest, sk_svc, and sk_gradient_boosting."""
from core.node import BaseNode, PortType

_VALID = {"logistic_regression", "random_forest", "svc", "gradient_boosting"}


class SkClassifierNode(BaseNode):
    type_name = "sk_classifier"
    label = "Classifier"
    category = "Sklearn"
    description = (
        "Fit a sklearn classifier. Select algorithm via the 'algorithm' input: "
        "logistic_regression, random_forest, svc, gradient_boosting."
    )

    def _setup_ports(self):
        self.add_input("algorithm",     PortType.STRING, "logistic_regression")
        self.add_input("X_train",       PortType.NDARRAY)
        self.add_input("y_train",       PortType.NDARRAY)
        # logistic_regression
        self.add_input("max_iter",      PortType.INT,   1000)
        # random_forest, gradient_boosting
        self.add_input("n_estimators",  PortType.INT,   100)
        # gradient_boosting
        self.add_input("learning_rate", PortType.FLOAT, 0.1)
        # svc
        self.add_input("C",             PortType.FLOAT, 1.0)
        self.add_input("kernel",        PortType.STRING, "rbf")
        # random_forest
        self.add_input("random_state",  PortType.INT,   42)
        self.add_output("model",        PortType.SKLEARN_MODEL)

    def execute(self, inputs):
        try:
            algo = (inputs.get("algorithm") or "logistic_regression").strip()
            X = inputs.get("X_train")
            y = inputs.get("y_train")
            if X is None or y is None:
                return {"model": None}

            if algo == "logistic_regression":
                from sklearn.linear_model import LogisticRegression
                m = LogisticRegression(
                    max_iter=int(inputs.get("max_iter", 1000)),
                )
            elif algo == "random_forest":
                from sklearn.ensemble import RandomForestClassifier
                m = RandomForestClassifier(
                    n_estimators=int(inputs.get("n_estimators", 100)),
                    random_state=int(inputs.get("random_state", 42)),
                )
            elif algo == "svc":
                from sklearn.svm import SVC
                m = SVC(
                    C=float(inputs.get("C", 1.0)),
                    kernel=inputs.get("kernel") or "rbf",
                )
            elif algo == "gradient_boosting":
                from sklearn.ensemble import GradientBoostingClassifier
                m = GradientBoostingClassifier(
                    n_estimators=int(inputs.get("n_estimators", 100)),
                    learning_rate=float(inputs.get("learning_rate", 0.1)),
                )
            else:
                return {"model": None}

            m.fit(X, y)
            return {"model": m}
        except Exception:
            return {"model": None}

    def export(self, iv, ov):
        algo = (self._val(iv, "algorithm") or "logistic_regression").strip().strip('"').strip("'")
        X = self._val(iv, "X_train")
        y = self._val(iv, "y_train")
        out = ov["model"]

        if algo == "logistic_regression":
            max_iter = self._val(iv, "max_iter")
            return (
                ["from sklearn.linear_model import LogisticRegression"],
                [
                    f"{out} = LogisticRegression(max_iter={max_iter})",
                    f"{out}.fit({X}, {y})",
                ],
            )
        elif algo == "random_forest":
            n = self._val(iv, "n_estimators")
            rs = self._val(iv, "random_state")
            return (
                ["from sklearn.ensemble import RandomForestClassifier"],
                [
                    f"{out} = RandomForestClassifier(n_estimators={n}, random_state={rs})",
                    f"{out}.fit({X}, {y})",
                ],
            )
        elif algo == "svc":
            C = self._val(iv, "C")
            k = self._val(iv, "kernel")
            return (
                ["from sklearn.svm import SVC"],
                [
                    f"{out} = SVC(C={C}, kernel={k})",
                    f"{out}.fit({X}, {y})",
                ],
            )
        elif algo == "gradient_boosting":
            n = self._val(iv, "n_estimators")
            lr = self._val(iv, "learning_rate")
            return (
                ["from sklearn.ensemble import GradientBoostingClassifier"],
                [
                    f"{out} = GradientBoostingClassifier(n_estimators={n}, learning_rate={lr})",
                    f"{out}.fit({X}, {y})",
                ],
            )
        else:
            return ([], [f"# sk_classifier: unknown algorithm '{algo}'"])
