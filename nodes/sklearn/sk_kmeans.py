"""KMeans clustering — fit_predict node."""
from core.node import BaseNode, PortType


class SkKMeansNode(BaseNode):
    type_name = "sk_kmeans"
    label = "KMeans"
    category = "Sklearn"
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

    def export(self, iv, ov):
        k = self._val(iv, "n_clusters")
        rs = self._val(iv, "random_state")
        X = self._val(iv, "X")
        return (
            ["from sklearn.cluster import KMeans"],
            [
                f"{ov['model']} = KMeans(n_clusters={k}, random_state={rs}, n_init='auto')",
                f"{ov['labels']} = {ov['model']}.fit_predict({X})",
            ],
        )
