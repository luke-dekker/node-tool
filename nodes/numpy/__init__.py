"""NumPy nodes — array creation, math, transforms, linear algebra."""
from __future__ import annotations
from core.node import BaseNode
from core.node import PortType

CATEGORY = "NumPy"


# ── Array Creation ────────────────────────────────────────────────────────────

class NpArangeNode(BaseNode):
    type_name = "np_arange"
    label = "Arange"
    category = CATEGORY
    description = "np.arange(start, stop, step)"

    def _setup_ports(self):
        self.add_input("start", PortType.FLOAT, 0.0)
        self.add_input("stop",  PortType.FLOAT, 10.0)
        self.add_input("step",  PortType.FLOAT, 1.0)
        self.add_output("array", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            import numpy as np
            start = inputs.get("start", 0.0)
            stop  = inputs.get("stop",  10.0)
            step  = inputs.get("step",  1.0)
            if any(v is None for v in [start, stop, step]):
                return {"array": None}
            return {"array": np.arange(float(start), float(stop), float(step))}
        except Exception:
            return {"array": None}


class NpLinspaceNode(BaseNode):
    type_name = "np_linspace"
    label = "Linspace"
    category = CATEGORY
    description = "np.linspace(start, stop, num)"

    def _setup_ports(self):
        self.add_input("start", PortType.FLOAT, 0.0)
        self.add_input("stop",  PortType.FLOAT, 1.0)
        self.add_input("num",   PortType.INT,   50)
        self.add_output("array", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            import numpy as np
            start = inputs.get("start", 0.0)
            stop  = inputs.get("stop",  1.0)
            num   = inputs.get("num",   50)
            if any(v is None for v in [start, stop, num]):
                return {"array": None}
            return {"array": np.linspace(float(start), float(stop), int(num))}
        except Exception:
            return {"array": None}


class NpZerosNode(BaseNode):
    type_name = "np_zeros"
    label = "Zeros"
    category = CATEGORY
    description = "np.zeros with shape string, e.g. '3,4'"

    def _setup_ports(self):
        self.add_input("shape", PortType.STRING, "3,4")
        self.add_output("array", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            import numpy as np
            shape_str = inputs.get("shape", "3,4") or "3,4"
            shape = tuple(int(x) for x in shape_str.split(",") if x.strip())
            return {"array": np.zeros(shape)}
        except Exception:
            return {"array": None}


class NpOnesNode(BaseNode):
    type_name = "np_ones"
    label = "Ones"
    category = CATEGORY
    description = "np.ones with shape string, e.g. '3,4'"

    def _setup_ports(self):
        self.add_input("shape", PortType.STRING, "3,4")
        self.add_output("array", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            import numpy as np
            shape_str = inputs.get("shape", "3,4") or "3,4"
            shape = tuple(int(x) for x in shape_str.split(",") if x.strip())
            return {"array": np.ones(shape)}
        except Exception:
            return {"array": None}


class NpRandNode(BaseNode):
    type_name = "np_rand"
    label = "Random Uniform"
    category = CATEGORY
    description = "np.random.rand with shape string. seed>=0 to fix seed."

    def _setup_ports(self):
        self.add_input("shape", PortType.STRING, "3,4")
        self.add_input("seed",  PortType.INT,    -1)
        self.add_output("array", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            import numpy as np
            shape_str = inputs.get("shape", "3,4") or "3,4"
            seed = inputs.get("seed", -1)
            shape = tuple(int(x) for x in shape_str.split(",") if x.strip())
            if seed is not None and int(seed) >= 0:
                np.random.seed(int(seed))
            return {"array": np.random.rand(*shape)}
        except Exception:
            return {"array": None}


class NpRandnNode(BaseNode):
    type_name = "np_randn"
    label = "Random Normal"
    category = CATEGORY
    description = "np.random.randn with shape string. seed>=0 to fix seed."

    def _setup_ports(self):
        self.add_input("shape", PortType.STRING, "3,4")
        self.add_input("seed",  PortType.INT,    -1)
        self.add_output("array", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            import numpy as np
            shape_str = inputs.get("shape", "3,4") or "3,4"
            seed = inputs.get("seed", -1)
            shape = tuple(int(x) for x in shape_str.split(",") if x.strip())
            if seed is not None and int(seed) >= 0:
                np.random.seed(int(seed))
            return {"array": np.random.randn(*shape)}
        except Exception:
            return {"array": None}


class NpFromListNode(BaseNode):
    type_name = "np_from_list"
    label = "From List"
    category = CATEGORY
    description = "Parse comma-separated values string into ndarray."

    def _setup_ports(self):
        self.add_input("values", PortType.STRING, "1,2,3,4")
        self.add_input("dtype",  PortType.STRING, "float32")
        self.add_output("array", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            import numpy as np
            values_str = inputs.get("values", "1,2,3,4") or ""
            dtype_str  = inputs.get("dtype",  "float32") or "float32"
            vals = [float(x) for x in values_str.split(",") if x.strip()]
            return {"array": np.array(vals, dtype=dtype_str)}
        except Exception:
            return {"array": None}


class NpEyeNode(BaseNode):
    type_name = "np_eye"
    label = "Eye"
    category = CATEGORY
    description = "np.eye(n) — identity matrix"

    def _setup_ports(self):
        self.add_input("n", PortType.INT, 3)
        self.add_output("array", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            import numpy as np
            n = inputs.get("n", 3)
            if n is None:
                return {"array": None}
            return {"array": np.eye(int(n))}
        except Exception:
            return {"array": None}


# ── Math & Stats ──────────────────────────────────────────────────────────────

class NpMeanNode(BaseNode):
    type_name = "np_mean"
    label = "Mean"
    category = CATEGORY
    description = "np.mean. axis=-99 means over all elements."

    def _setup_ports(self):
        self.add_input("array", PortType.NDARRAY)
        self.add_input("axis",  PortType.INT, -99)
        self.add_output("result", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            import numpy as np
            arr  = inputs.get("array")
            axis = inputs.get("axis", -99)
            if arr is None:
                return {"result": None}
            ax = None if (axis is None or int(axis) == -99) else int(axis)
            return {"result": np.mean(arr, axis=ax)}
        except Exception:
            return {"result": None}


class NpStdNode(BaseNode):
    type_name = "np_std"
    label = "Std Dev"
    category = CATEGORY
    description = "np.std. axis=-99 means over all elements."

    def _setup_ports(self):
        self.add_input("array", PortType.NDARRAY)
        self.add_input("axis",  PortType.INT, -99)
        self.add_output("result", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            import numpy as np
            arr  = inputs.get("array")
            axis = inputs.get("axis", -99)
            if arr is None:
                return {"result": None}
            ax = None if (axis is None or int(axis) == -99) else int(axis)
            return {"result": np.std(arr, axis=ax)}
        except Exception:
            return {"result": None}


class NpSumNode(BaseNode):
    type_name = "np_sum"
    label = "Sum"
    category = CATEGORY
    description = "np.sum. axis=-99 means over all elements."

    def _setup_ports(self):
        self.add_input("array", PortType.NDARRAY)
        self.add_input("axis",  PortType.INT, -99)
        self.add_output("result", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            import numpy as np
            arr  = inputs.get("array")
            axis = inputs.get("axis", -99)
            if arr is None:
                return {"result": None}
            ax = None if (axis is None or int(axis) == -99) else int(axis)
            return {"result": np.sum(arr, axis=ax)}
        except Exception:
            return {"result": None}


class NpMinNode(BaseNode):
    type_name = "np_min"
    label = "Min"
    category = CATEGORY
    description = "np.min(array)"

    def _setup_ports(self):
        self.add_input("array", PortType.NDARRAY)
        self.add_output("result", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            import numpy as np
            arr = inputs.get("array")
            if arr is None:
                return {"result": None}
            return {"result": np.min(arr)}
        except Exception:
            return {"result": None}


class NpMaxNode(BaseNode):
    type_name = "np_max"
    label = "Max"
    category = CATEGORY
    description = "np.max(array)"

    def _setup_ports(self):
        self.add_input("array", PortType.NDARRAY)
        self.add_output("result", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            import numpy as np
            arr = inputs.get("array")
            if arr is None:
                return {"result": None}
            return {"result": np.max(arr)}
        except Exception:
            return {"result": None}


class NpAbsNode(BaseNode):
    type_name = "np_abs"
    label = "Abs"
    category = CATEGORY
    description = "np.abs(array)"

    def _setup_ports(self):
        self.add_input("array", PortType.NDARRAY)
        self.add_output("result", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            import numpy as np
            arr = inputs.get("array")
            if arr is None:
                return {"result": None}
            return {"result": np.abs(arr)}
        except Exception:
            return {"result": None}


class NpSqrtNode(BaseNode):
    type_name = "np_sqrt"
    label = "Sqrt"
    category = CATEGORY
    description = "np.sqrt(array)"

    def _setup_ports(self):
        self.add_input("array", PortType.NDARRAY)
        self.add_output("result", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            import numpy as np
            arr = inputs.get("array")
            if arr is None:
                return {"result": None}
            return {"result": np.sqrt(arr)}
        except Exception:
            return {"result": None}


class NpLogNode(BaseNode):
    type_name = "np_log"
    label = "Log"
    category = CATEGORY
    description = "np.log(array) — clipped to avoid -inf"

    def _setup_ports(self):
        self.add_input("array", PortType.NDARRAY)
        self.add_output("result", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            import numpy as np
            arr = inputs.get("array")
            if arr is None:
                return {"result": None}
            return {"result": np.log(np.clip(arr, 1e-12, None))}
        except Exception:
            return {"result": None}


class NpExpNode(BaseNode):
    type_name = "np_exp"
    label = "Exp"
    category = CATEGORY
    description = "np.exp(array)"

    def _setup_ports(self):
        self.add_input("array", PortType.NDARRAY)
        self.add_output("result", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            import numpy as np
            arr = inputs.get("array")
            if arr is None:
                return {"result": None}
            return {"result": np.exp(arr)}
        except Exception:
            return {"result": None}


class NpClipNode(BaseNode):
    type_name = "np_clip"
    label = "Clip"
    category = CATEGORY
    description = "np.clip(array, a_min, a_max)"

    def _setup_ports(self):
        self.add_input("array", PortType.NDARRAY)
        self.add_input("a_min", PortType.FLOAT, 0.0)
        self.add_input("a_max", PortType.FLOAT, 1.0)
        self.add_output("result", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            import numpy as np
            arr   = inputs.get("array")
            a_min = inputs.get("a_min", 0.0)
            a_max = inputs.get("a_max", 1.0)
            if arr is None:
                return {"result": None}
            return {"result": np.clip(arr, float(a_min), float(a_max))}
        except Exception:
            return {"result": None}


class NpNormalizeNode(BaseNode):
    type_name = "np_normalize"
    label = "Normalize"
    category = CATEGORY
    description = "(x - min)/(max - min) per array"

    def _setup_ports(self):
        self.add_input("array", PortType.NDARRAY)
        self.add_output("result", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            import numpy as np
            arr = inputs.get("array")
            if arr is None:
                return {"result": None}
            mn = arr.min()
            mx = arr.max()
            rng = mx - mn
            if rng == 0:
                return {"result": np.zeros_like(arr, dtype=float)}
            return {"result": (arr - mn) / rng}
        except Exception:
            return {"result": None}


# ── Array Ops ─────────────────────────────────────────────────────────────────

class NpReshapeNode(BaseNode):
    type_name = "np_reshape"
    label = "Reshape"
    category = CATEGORY
    description = "Reshape array to given shape string, e.g. '2,3'"

    def _setup_ports(self):
        self.add_input("array", PortType.NDARRAY)
        self.add_input("shape", PortType.STRING, "2,3")
        self.add_output("result", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            import numpy as np
            arr       = inputs.get("array")
            shape_str = inputs.get("shape", "2,3") or "2,3"
            if arr is None:
                return {"result": None}
            shape = tuple(int(x) for x in shape_str.split(",") if x.strip())
            return {"result": arr.reshape(shape)}
        except Exception:
            return {"result": None}


class NpTransposeNode(BaseNode):
    type_name = "np_transpose"
    label = "Transpose"
    category = CATEGORY
    description = "np.transpose(array)"

    def _setup_ports(self):
        self.add_input("array", PortType.NDARRAY)
        self.add_output("result", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            import numpy as np
            arr = inputs.get("array")
            if arr is None:
                return {"result": None}
            return {"result": np.transpose(arr)}
        except Exception:
            return {"result": None}


class NpFlattenNode(BaseNode):
    type_name = "np_flatten"
    label = "Flatten"
    category = CATEGORY
    description = "array.flatten()"

    def _setup_ports(self):
        self.add_input("array", PortType.NDARRAY)
        self.add_output("result", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            arr = inputs.get("array")
            if arr is None:
                return {"result": None}
            return {"result": arr.flatten()}
        except Exception:
            return {"result": None}


class NpConcatNode(BaseNode):
    type_name = "np_concat"
    label = "Concatenate"
    category = CATEGORY
    description = "np.concatenate([a, b], axis=axis)"

    def _setup_ports(self):
        self.add_input("a",    PortType.NDARRAY)
        self.add_input("b",    PortType.NDARRAY)
        self.add_input("axis", PortType.INT, 0)
        self.add_output("result", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            import numpy as np
            a    = inputs.get("a")
            b    = inputs.get("b")
            axis = inputs.get("axis", 0)
            if a is None or b is None:
                return {"result": None}
            return {"result": np.concatenate([a, b], axis=int(axis))}
        except Exception:
            return {"result": None}


class NpStackNode(BaseNode):
    type_name = "np_stack"
    label = "Stack"
    category = CATEGORY
    description = "np.stack([a, b], axis=axis)"

    def _setup_ports(self):
        self.add_input("a",    PortType.NDARRAY)
        self.add_input("b",    PortType.NDARRAY)
        self.add_input("axis", PortType.INT, 0)
        self.add_output("result", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            import numpy as np
            a    = inputs.get("a")
            b    = inputs.get("b")
            axis = inputs.get("axis", 0)
            if a is None or b is None:
                return {"result": None}
            return {"result": np.stack([a, b], axis=int(axis))}
        except Exception:
            return {"result": None}


class NpSliceNode(BaseNode):
    type_name = "np_slice"
    label = "Slice"
    category = CATEGORY
    description = "array[start:end:step] along first axis"

    def _setup_ports(self):
        self.add_input("array", PortType.NDARRAY)
        self.add_input("start", PortType.INT, 0)
        self.add_input("end",   PortType.INT, 10)
        self.add_input("step",  PortType.INT, 1)
        self.add_output("result", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            arr   = inputs.get("array")
            start = inputs.get("start", 0)
            end   = inputs.get("end",   10)
            step  = inputs.get("step",  1)
            if arr is None:
                return {"result": None}
            return {"result": arr[int(start):int(end):int(step)]}
        except Exception:
            return {"result": None}


class NpWhereNode(BaseNode):
    type_name = "np_where"
    label = "Where"
    category = CATEGORY
    description = "np.where(condition>0, x, y)"

    def _setup_ports(self):
        self.add_input("condition", PortType.NDARRAY)
        self.add_input("x",         PortType.NDARRAY)
        self.add_input("y",         PortType.NDARRAY)
        self.add_output("result",   PortType.NDARRAY)

    def execute(self, inputs):
        try:
            import numpy as np
            cond = inputs.get("condition")
            x    = inputs.get("x")
            y    = inputs.get("y")
            if any(v is None for v in [cond, x, y]):
                return {"result": None}
            return {"result": np.where(cond > 0, x, y)}
        except Exception:
            return {"result": None}


# ── Linear Algebra ────────────────────────────────────────────────────────────

class NpDotNode(BaseNode):
    type_name = "np_dot"
    label = "Dot"
    category = CATEGORY
    description = "np.dot(a, b)"

    def _setup_ports(self):
        self.add_input("a", PortType.NDARRAY)
        self.add_input("b", PortType.NDARRAY)
        self.add_output("result", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            import numpy as np
            a = inputs.get("a")
            b = inputs.get("b")
            if a is None or b is None:
                return {"result": None}
            return {"result": np.dot(a, b)}
        except Exception:
            return {"result": None}


class NpMatMulNode(BaseNode):
    type_name = "np_matmul"
    label = "MatMul"
    category = CATEGORY
    description = "np.matmul(a, b)"

    def _setup_ports(self):
        self.add_input("a", PortType.NDARRAY)
        self.add_input("b", PortType.NDARRAY)
        self.add_output("result", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            import numpy as np
            a = inputs.get("a")
            b = inputs.get("b")
            if a is None or b is None:
                return {"result": None}
            return {"result": np.matmul(a, b)}
        except Exception:
            return {"result": None}


class NpInvNode(BaseNode):
    type_name = "np_inv"
    label = "Inverse"
    category = CATEGORY
    description = "np.linalg.inv(matrix)"

    def _setup_ports(self):
        self.add_input("matrix", PortType.NDARRAY)
        self.add_output("result", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            import numpy as np
            mat = inputs.get("matrix")
            if mat is None:
                return {"result": None}
            return {"result": np.linalg.inv(mat)}
        except Exception:
            return {"result": None}


class NpEigNode(BaseNode):
    type_name = "np_eig"
    label = "Eigenvalues"
    category = CATEGORY
    description = "np.linalg.eig(matrix) — eigenvalues and eigenvectors"

    def _setup_ports(self):
        self.add_input("matrix",      PortType.NDARRAY)
        self.add_output("eigenvalues",  PortType.NDARRAY)
        self.add_output("eigenvectors", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            import numpy as np
            mat = inputs.get("matrix")
            if mat is None:
                return {"eigenvalues": None, "eigenvectors": None}
            vals, vecs = np.linalg.eig(mat)
            return {"eigenvalues": vals, "eigenvectors": vecs}
        except Exception:
            return {"eigenvalues": None, "eigenvectors": None}


class NpSVDNode(BaseNode):
    type_name = "np_svd"
    label = "SVD"
    category = CATEGORY
    description = "np.linalg.svd — U, S, Vt"

    def _setup_ports(self):
        self.add_input("matrix", PortType.NDARRAY)
        self.add_output("U",  PortType.NDARRAY)
        self.add_output("S",  PortType.NDARRAY)
        self.add_output("Vt", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            import numpy as np
            mat = inputs.get("matrix")
            if mat is None:
                return {"U": None, "S": None, "Vt": None}
            U, S, Vt = np.linalg.svd(mat, full_matrices=False)
            return {"U": U, "S": S, "Vt": Vt}
        except Exception:
            return {"U": None, "S": None, "Vt": None}


# ── Info ──────────────────────────────────────────────────────────────────────

class NpArrayInfoNode(BaseNode):
    type_name = "np_array_info"
    label = "Array Info"
    category = CATEGORY
    description = "Print shape, dtype, min, max, mean of array"

    def _setup_ports(self):
        self.add_input("array", PortType.NDARRAY)
        self.add_output("info", PortType.STRING)

    def execute(self, inputs):
        try:
            arr = inputs.get("array")
            if arr is None:
                return {"info": "None"}
            import numpy as np
            a = np.asarray(arr, dtype=float)
            info = (f"shape={list(arr.shape)} dtype={arr.dtype} "
                    f"min={a.min():.4g} max={a.max():.4g} mean={a.mean():.4g}")
            return {"info": info}
        except Exception:
            return {"info": "error"}


class NpShapeNode(BaseNode):
    type_name = "np_shape"
    label = "Shape"
    category = CATEGORY
    description = "Return array shape as string"

    def _setup_ports(self):
        self.add_input("array",  PortType.NDARRAY)
        self.add_output("shape", PortType.STRING)

    def execute(self, inputs):
        try:
            arr = inputs.get("array")
            if arr is None:
                return {"shape": "None"}
            return {"shape": str(list(arr.shape))}
        except Exception:
            return {"shape": "error"}
