"""PCA dimensionality reduction — fit_transform node."""
from core.node import BaseNode, PortType


class SkPCANode(BaseNode):
    type_name = "sk_pca"
    label = "PCA"
    category = "Sklearn"
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

    def export(self, iv, ov):
        n = self._val(iv, "n_components")
        X = self._val(iv, "X")
        return (
            ["from sklearn.decomposition import PCA"],
            [
                f"{ov['model']} = PCA(n_components={n})",
                f"{ov['transformed']} = {ov['model']}.fit_transform({X})",
            ],
        )
