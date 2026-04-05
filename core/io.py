"""Graph I/O -- JSON serialization and Python script export."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Callable
from core.graph import Graph
from core.node import BaseNode
from nodes import NODE_REGISTRY


# ── Serializer (save/load graph to JSON) ──────────────────────────────────────



class Serializer:
    VERSION = 1

    SERIALIZABLE_TYPES = (int, float, bool, str, type(None))

    @staticmethod
    def _serialize_value(v):
        if isinstance(v, Serializer.SERIALIZABLE_TYPES):
            return v
        return None  # tensors, modules etc → null

    @classmethod
    def save(cls, graph: Graph, positions: dict[str, tuple], path: str) -> None:
        nodes = []
        for node_id, node in graph.nodes.items():
            nodes.append({
                "id": node.id,
                "type_name": node.type_name,
                "pos": list(positions.get(node_id, [100, 100])),
                "inputs": {
                    k: cls._serialize_value(p.default_value)
                    for k, p in node.inputs.items()
                }
            })
        connections = [
            {
                "from_node": c.from_node_id,
                "from_port": c.from_port,
                "to_node": c.to_node_id,
                "to_port": c.to_port,
            }
            for c in graph.connections
        ]
        data = {"version": cls.VERSION, "nodes": nodes, "connections": connections}
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str) -> tuple[Graph, dict[str, list]]:
        data = json.loads(Path(path).read_text())
        graph = Graph()
        positions = {}
        for nd in data["nodes"]:
            cls_ = NODE_REGISTRY.get(nd["type_name"])
            if cls_ is None:
                raise KeyError(f"Unknown node type: {nd['type_name']}")
            node = cls_()
            # Reuse saved ID
            node.id = nd["id"]
            # Restore input defaults
            for k, v in nd.get("inputs", {}).items():
                if k in node.inputs and v is not None:
                    node.inputs[k].default_value = v
            graph.add_node(node)
            positions[node.id] = nd.get("pos", [100, 100])
        for c in data.get("connections", []):
            graph.add_connection(c["from_node"], c["from_port"], c["to_node"], c["to_port"])
        return graph, positions


# ── Exporter (graph -> Python script) ───────────────────────────────────────


# ── Helpers ───────────────────────────────────────────────────────────────────

_PREFIX_STRIP = ("pt_", "np_", "pd_", "sk_", "sp_", "viz_")

def _short(type_name: str) -> str:
    """Turn a type_name into a clean variable prefix."""
    s = type_name
    for p in _PREFIX_STRIP:
        if s.startswith(p):
            s = s[len(p):]
            break
    # Remove remaining underscores-only segments for readability
    s = s.replace("__terminal__", "terminal")
    return s or type_name


def _val(node: BaseNode, in_vars: dict[str, str | None], port_name: str) -> str:
    """Return a Python expression: connected variable name OR literal default."""
    v = in_vars.get(port_name)
    if v is not None:
        return v
    default = node.inputs[port_name].default_value
    if isinstance(default, bool):
        return "True" if default else "False"
    if isinstance(default, str):
        return repr(default)
    if default is None:
        return "None"
    return repr(default)


def _axis(node: BaseNode, in_vars: dict, port_name: str = "axis") -> str:
    """Translate axis=-99 sentinel → None, else use value."""
    v = in_vars.get(port_name)
    if v is not None:
        return v
    default = node.inputs[port_name].default_value
    if default is None or (isinstance(default, int) and default == -99):
        return "None"
    return repr(default)


# ── Generator registry ────────────────────────────────────────────────────────
# Each generator: fn(node, in_vars, out_vars) -> (imports: list[str], lines: list[str])
# in_vars[port_name]  = Python var name if connected, else None
# out_vars[port_name] = Python var name to assign this output to

GeneratorFn = Callable[[BaseNode, dict, dict], tuple[list[str], list[str]]]
_GENERATORS: dict[str, GeneratorFn] = {}


def _reg(*type_names: str):
    """Decorator to register a generator for one or more type_names."""
    def decorator(fn: GeneratorFn) -> GeneratorFn:
        for name in type_names:
            _GENERATORS[name] = fn
        return fn
    return decorator


# ── Math ──────────────────────────────────────────────────────────────────────

@_reg("add")
def _gen_add(node, iv, ov):
    return [], [f"{ov['Result']} = {_val(node,iv,'A')} + {_val(node,iv,'B')}"]

@_reg("subtract")
def _gen_sub(node, iv, ov):
    return [], [f"{ov['Result']} = {_val(node,iv,'A')} - {_val(node,iv,'B')}"]

@_reg("multiply")
def _gen_mul(node, iv, ov):
    return [], [f"{ov['Result']} = {_val(node,iv,'A')} * {_val(node,iv,'B')}"]

@_reg("divide")
def _gen_div(node, iv, ov):
    b = _val(node, iv, 'B')
    return [], [f"{ov['Result']} = {_val(node,iv,'A')} / {b} if {b} != 0 else 0.0"]

@_reg("power")
def _gen_pow(node, iv, ov):
    return [], [f"{ov['Result']} = {_val(node,iv,'Base')} ** {_val(node,iv,'Exp')}"]

@_reg("clamp")
def _gen_clamp(node, iv, ov):
    v, lo, hi = _val(node,iv,'Value'), _val(node,iv,'Min'), _val(node,iv,'Max')
    return [], [f"{ov['Result']} = max({lo}, min({hi}, {v}))"]

@_reg("map_range")
def _gen_maprange(node, iv, ov):
    v = _val(node,iv,'Value'); i0=_val(node,iv,'InMin'); i1=_val(node,iv,'InMax')
    o0=_val(node,iv,'OutMin'); o1=_val(node,iv,'OutMax')
    return [], [
        f"_denom = {i1} - {i0}",
        f"{ov['Result']} = {o0} + ({v} - {i0}) / _denom * ({o1} - {o0}) if _denom != 0 else {o0}",
    ]

@_reg("round")
def _gen_round(node, iv, ov):
    return [], [f"{ov['Result']} = round({_val(node,iv,'Value')}, {_val(node,iv,'Decimals')})"]

@_reg("abs")
def _gen_abs(node, iv, ov):
    return [], [f"{ov['Result']} = abs({_val(node,iv,'Value')})"]

@_reg("sqrt")
def _gen_sqrt(node, iv, ov):
    return ["import math"], [f"{ov['Result']} = math.sqrt(max(0, {_val(node,iv,'Value')}))"]

@_reg("sin")
def _gen_sin(node, iv, ov):
    return ["import math"], [f"{ov['Result']} = math.sin(math.radians({_val(node,iv,'Degrees')}))"]

@_reg("cos")
def _gen_cos(node, iv, ov):
    return ["import math"], [f"{ov['Result']} = math.cos(math.radians({_val(node,iv,'Degrees')}))"]

@_reg("random_float")
def _gen_randf(node, iv, ov):
    lo, hi = _val(node,iv,'Min'), _val(node,iv,'Max')
    return ["import random"], [f"{ov['Result']} = random.uniform({lo}, {hi})"]


# ── Logic ─────────────────────────────────────────────────────────────────────

_CMP_OPS = ["<", "<=", "==", ">=", ">", "!="]

@_reg("compare")
def _gen_compare(node, iv, ov):
    a, b = _val(node,iv,'A'), _val(node,iv,'B')
    op_default = int(node.inputs['Op'].default_value or 2)
    op_var = iv.get('Op')
    if op_var:
        # runtime — generate a runtime branch
        ops_str = repr(_CMP_OPS)
        return [], [
            f"_cmp_ops = {ops_str}",
            f"_cmp_fn = lambda x,y,o: eval(f'{{x}} {{_cmp_ops[o%6]}} {{y}}')",
            f"{ov['Result']} = _cmp_fn({a}, {b}, int({op_var}))",
        ]
    else:
        op = _CMP_OPS[op_default % 6]
        return [], [f"{ov['Result']} = {a} {op} {b}"]

@_reg("and")
def _gen_and(node, iv, ov):
    return [], [f"{ov['Result']} = bool({_val(node,iv,'A')}) and bool({_val(node,iv,'B')})"]

@_reg("or")
def _gen_or(node, iv, ov):
    return [], [f"{ov['Result']} = bool({_val(node,iv,'A')}) or bool({_val(node,iv,'B')})"]

@_reg("not")
def _gen_not(node, iv, ov):
    return [], [f"{ov['Result']} = not bool({_val(node,iv,'Value')})"]

@_reg("branch")
def _gen_branch(node, iv, ov):
    cond = _val(node,iv,'Condition')
    tv = _val(node,iv,'True Value'); fv = _val(node,iv,'False Value')
    return [], [f"{ov['Result']} = {tv} if {cond} else {fv}"]


# ── String ────────────────────────────────────────────────────────────────────

@_reg("concat")
def _gen_concat(node, iv, ov):
    return [], [f"{ov['Result']} = str({_val(node,iv,'A')}) + str({_val(node,iv,'B')})"]

@_reg("format")
def _gen_format(node, iv, ov):
    t = _val(node,iv,'Template'); a0=_val(node,iv,'Arg0'); a1=_val(node,iv,'Arg1')
    return [], [f"{ov['Result']} = str({t}).format({a0}, {a1})"]

@_reg("upper")
def _gen_upper(node, iv, ov):
    return [], [f"{ov['Result']} = str({_val(node,iv,'Value')}).upper()"]

@_reg("lower")
def _gen_lower(node, iv, ov):
    return [], [f"{ov['Result']} = str({_val(node,iv,'Value')}).lower()"]

@_reg("strip")
def _gen_strip(node, iv, ov):
    return [], [f"{ov['Result']} = str({_val(node,iv,'Value')}).strip()"]

@_reg("length")
def _gen_length(node, iv, ov):
    return [], [f"{ov['Length']} = len(str({_val(node,iv,'Value')}))"]

@_reg("contains")
def _gen_contains(node, iv, ov):
    return [], [f"{ov['Result']} = str({_val(node,iv,'Needle')}) in str({_val(node,iv,'Haystack')})"]

@_reg("replace")
def _gen_replace(node, iv, ov):
    v=_val(node,iv,'Value'); old=_val(node,iv,'Old'); new=_val(node,iv,'New')
    return [], [f"{ov['Result']} = str({v}).replace(str({old}), str({new}))"]


# ── Consolidated nodes (MathNode, LogicNode, StringNode, PythonNode) ──────────

_MATH_OP_EXPRS = {
    "add":      lambda a, b: f"{a} + {b}",
    "subtract": lambda a, b: f"{a} - {b}",
    "multiply": lambda a, b: f"{a} * {b}",
    "divide":   lambda a, b: f"{a} / {b} if {b} != 0 else 0.0",
    "power":    lambda a, b: f"float({a} ** {b})",
    "sqrt":     lambda a, b: f"math.sqrt(max(0.0, {a}))",
    "abs":      lambda a, b: f"abs({a})",
    "sin":      lambda a, b: f"math.sin(math.radians({a}))",
    "cos":      lambda a, b: f"math.cos(math.radians({a}))",
    "tan":      lambda a, b: f"math.tan(math.radians({a}))",
    "round":    lambda a, b: f"float(round({a}, int({b})))",
    "random":   lambda a, b: f"random.uniform(min({a},{b}), max({a},{b}))",
}

_MATH_OP_IMPORTS = {
    "sqrt": ["import math"], "sin": ["import math"],
    "cos":  ["import math"], "tan": ["import math"],
    "random": ["import random"],
}

@_reg("math")
def _gen_math(node, iv, ov):
    op = str(node.inputs["Op"].default_value or "add").strip().lower()
    if iv.get("Op"):
        # Op wired at runtime — fall back to safe eval dispatch
        a, b = _val(node,iv,"A"), _val(node,iv,"B")
        return ["import math", "import random"], [
            f"{ov['Result']} = eval(str({iv['Op']}).strip().lower()+"
            f"'({a},{b})')",  # not ideal, but honest fallback
        ]
    expr_fn = _MATH_OP_EXPRS.get(op, _MATH_OP_EXPRS["add"])
    imps = _MATH_OP_IMPORTS.get(op, [])
    a, b = _val(node,iv,"A"), _val(node,iv,"B")
    return imps, [f"{ov['Result']} = {expr_fn(a, b)}"]


_LOGIC_OP_EXPRS = {
    "and": lambda a, b: f"bool({a}) and bool({b})",
    "or":  lambda a, b: f"bool({a}) or bool({b})",
    "not": lambda a, b: f"not bool({a})",
    "eq":  lambda a, b: f"{a} == {b}",
    "neq": lambda a, b: f"{a} != {b}",
    "lt":  lambda a, b: f"{a} < {b}",
    "lte": lambda a, b: f"{a} <= {b}",
    "gt":  lambda a, b: f"{a} > {b}",
    "gte": lambda a, b: f"{a} >= {b}",
}

@_reg("logic")
def _gen_logic(node, iv, ov):
    op = str(node.inputs["Op"].default_value or "and").strip().lower()
    expr_fn = _LOGIC_OP_EXPRS.get(op, _LOGIC_OP_EXPRS["and"])
    a, b = _val(node,iv,"A"), _val(node,iv,"B")
    return [], [f"{ov['Result']} = {expr_fn(a, b)}"]


_STRING_OP_EXPRS = {
    "upper":    lambda a, b: f"str({a}).upper()",
    "lower":    lambda a, b: f"str({a}).lower()",
    "strip":    lambda a, b: f"str({a}).strip()",
    "reverse":  lambda a, b: f"str({a})[::-1]",
    "length":   lambda a, b: f"len(str({a}))",
    "concat":   lambda a, b: f"str({a}) + str({b})",
    "contains": lambda a, b: f"str({b}) in str({a})",
    "repeat":   lambda a, b: f"str({a}) * max(0, int(float({b})))",
}

@_reg("string_op")
def _gen_string_op(node, iv, ov):
    op = str(node.inputs["Op"].default_value or "upper").strip().lower()
    expr_fn = _STRING_OP_EXPRS.get(op, _STRING_OP_EXPRS["upper"])
    a, b = _val(node,iv,"A"), _val(node,iv,"B")
    return [], [f"{ov['Result']} = {expr_fn(a, b)}"]


@_reg("python")
def _gen_python(node, iv, ov):
    code = node.inputs["code"].default_value or "result = None"
    a = _val(node,iv,"a"); b = _val(node,iv,"b"); c = _val(node,iv,"c")
    lines = [
        f"_py_ns = {{'a': {a}, 'b': {b}, 'c': {c}}}",
        f"import math as _m, re; _py_ns.update({{'math':_m,'re':re}})",
        f"exec({repr(code)}, _py_ns)",
        f"{ov['result']} = _py_ns.get('result')",
    ]
    return [], lines


# ── Data ──────────────────────────────────────────────────────────────────────

@_reg("float_const")
def _gen_float_const(node, iv, ov):
    return [], [f"{ov['Value']} = float({_val(node,iv,'Value')})"]

@_reg("int_const")
def _gen_int_const(node, iv, ov):
    return [], [f"{ov['Value']} = int({_val(node,iv,'Value')})"]

@_reg("bool_const")
def _gen_bool_const(node, iv, ov):
    return [], [f"{ov['Value']} = bool({_val(node,iv,'Value')})"]

@_reg("string_const")
def _gen_string_const(node, iv, ov):
    return [], [f"{ov['Value']} = str({_val(node,iv,'Value')})"]

@_reg("print")
def _gen_print(node, iv, ov):
    v = _val(node,iv,'Value'); lbl = _val(node,iv,'Label')
    lines = [
        f"{ov['Value']} = {v}",
        f"print(f\"{{{lbl}}}: {{{ov['Value']}}}\" if {lbl} else str({ov['Value']}))",
    ]
    return [], lines

@_reg("to_float")
def _gen_to_float(node, iv, ov):
    return [], [f"{ov['Result']} = float({_val(node,iv,'Value')})"]

@_reg("to_int")
def _gen_to_int(node, iv, ov):
    return [], [f"{ov['Result']} = int(float({_val(node,iv,'Value')}))"]

@_reg("to_string")
def _gen_to_string(node, iv, ov):
    return [], [f"{ov['Result']} = str({_val(node,iv,'Value')})"]

@_reg("to_bool")
def _gen_to_bool(node, iv, ov):
    return [], [f"{ov['Result']} = bool({_val(node,iv,'Value')})"]


# ── NumPy ─────────────────────────────────────────────────────────────────────

@_reg("np_arange")
def _gen_np_arange(node, iv, ov):
    return ["import numpy as np"], [
        f"{ov['array']} = np.arange({_val(node,iv,'start')}, {_val(node,iv,'stop')}, {_val(node,iv,'step')})"
    ]

@_reg("np_linspace")
def _gen_np_linspace(node, iv, ov):
    return ["import numpy as np"], [
        f"{ov['array']} = np.linspace({_val(node,iv,'start')}, {_val(node,iv,'stop')}, {_val(node,iv,'num')})"
    ]

@_reg("np_zeros")
def _gen_np_zeros(node, iv, ov):
    shape = _val(node,iv,'shape')
    return ["import numpy as np"], [
        f"{ov['array']} = np.zeros(tuple(int(x) for x in {shape}.split(',') if x.strip()))"
    ]

@_reg("np_ones")
def _gen_np_ones(node, iv, ov):
    shape = _val(node,iv,'shape')
    return ["import numpy as np"], [
        f"{ov['array']} = np.ones(tuple(int(x) for x in {shape}.split(',') if x.strip()))"
    ]

@_reg("np_rand")
def _gen_np_rand(node, iv, ov):
    shape = _val(node,iv,'shape'); seed = _val(node,iv,'seed')
    return ["import numpy as np"], [
        f"_shape = tuple(int(x) for x in {shape}.split(',') if x.strip())",
        f"if int({seed}) >= 0: np.random.seed(int({seed}))",
        f"{ov['array']} = np.random.rand(*_shape)",
    ]

@_reg("np_randn")
def _gen_np_randn(node, iv, ov):
    shape = _val(node,iv,'shape'); seed = _val(node,iv,'seed')
    return ["import numpy as np"], [
        f"_shape = tuple(int(x) for x in {shape}.split(',') if x.strip())",
        f"if int({seed}) >= 0: np.random.seed(int({seed}))",
        f"{ov['array']} = np.random.randn(*_shape)",
    ]

@_reg("np_from_list")
def _gen_np_from_list(node, iv, ov):
    vals = _val(node,iv,'values'); dtype = _val(node,iv,'dtype')
    return ["import numpy as np"], [
        f"{ov['array']} = np.array([float(x) for x in {vals}.split(',') if x.strip()], dtype={dtype})"
    ]

@_reg("np_eye")
def _gen_np_eye(node, iv, ov):
    return ["import numpy as np"], [f"{ov['array']} = np.eye({_val(node,iv,'n')})"]

@_reg("np_mean")
def _gen_np_mean(node, iv, ov):
    ax = _axis(node, iv)
    return ["import numpy as np"], [f"{ov['result']} = np.mean({_val(node,iv,'array')}, axis={ax})"]

@_reg("np_std")
def _gen_np_std(node, iv, ov):
    ax = _axis(node, iv)
    return ["import numpy as np"], [f"{ov['result']} = np.std({_val(node,iv,'array')}, axis={ax})"]

@_reg("np_sum")
def _gen_np_sum(node, iv, ov):
    ax = _axis(node, iv)
    return ["import numpy as np"], [f"{ov['result']} = np.sum({_val(node,iv,'array')}, axis={ax})"]

@_reg("np_min")
def _gen_np_min(node, iv, ov):
    return ["import numpy as np"], [f"{ov['result']} = np.min({_val(node,iv,'array')})"]

@_reg("np_max")
def _gen_np_max(node, iv, ov):
    return ["import numpy as np"], [f"{ov['result']} = np.max({_val(node,iv,'array')})"]

@_reg("np_abs")
def _gen_np_abs(node, iv, ov):
    return ["import numpy as np"], [f"{ov['result']} = np.abs({_val(node,iv,'array')})"]

@_reg("np_sqrt")
def _gen_np_sqrt(node, iv, ov):
    return ["import numpy as np"], [f"{ov['result']} = np.sqrt({_val(node,iv,'array')})"]

@_reg("np_log")
def _gen_np_log(node, iv, ov):
    arr = _val(node, iv, 'array')
    return ["import numpy as np"], [f"{ov['result']} = np.log(np.clip({arr}, 1e-12, None))"]

@_reg("np_exp")
def _gen_np_exp(node, iv, ov):
    return ["import numpy as np"], [f"{ov['result']} = np.exp({_val(node,iv,'array')})"]

@_reg("np_clip")
def _gen_np_clip(node, iv, ov):
    return ["import numpy as np"], [
        f"{ov['result']} = np.clip({_val(node,iv,'array')}, {_val(node,iv,'a_min')}, {_val(node,iv,'a_max')})"
    ]

@_reg("np_normalize")
def _gen_np_normalize(node, iv, ov):
    arr = _val(node, iv, 'array')
    return ["import numpy as np"], [
        f"_mn, _mx = {arr}.min(), {arr}.max()",
        f"{ov['result']} = ({arr} - _mn) / (_mx - _mn) if _mx != _mn else np.zeros_like({arr}, dtype=float)",
    ]

@_reg("np_reshape")
def _gen_np_reshape(node, iv, ov):
    arr = _val(node,iv,'array'); shape = _val(node,iv,'shape')
    return ["import numpy as np"], [
        f"{ov['result']} = {arr}.reshape(tuple(int(x) for x in {shape}.split(',') if x.strip()))"
    ]

@_reg("np_transpose")
def _gen_np_transpose(node, iv, ov):
    return ["import numpy as np"], [f"{ov['result']} = np.transpose({_val(node,iv,'array')})"]

@_reg("np_flatten")
def _gen_np_flatten(node, iv, ov):
    return [], [f"{ov['result']} = {_val(node,iv,'array')}.flatten()"]

@_reg("np_concat")
def _gen_np_concat(node, iv, ov):
    return ["import numpy as np"], [
        f"{ov['result']} = np.concatenate([{_val(node,iv,'a')}, {_val(node,iv,'b')}], axis={_val(node,iv,'axis')})"
    ]

@_reg("np_stack")
def _gen_np_stack(node, iv, ov):
    return ["import numpy as np"], [
        f"{ov['result']} = np.stack([{_val(node,iv,'a')}, {_val(node,iv,'b')}], axis={_val(node,iv,'axis')})"
    ]

@_reg("np_slice")
def _gen_np_slice(node, iv, ov):
    arr=_val(node,iv,'array'); s=_val(node,iv,'start'); e=_val(node,iv,'end'); st=_val(node,iv,'step')
    return [], [f"{ov['result']} = {arr}[{s}:{e}:{st}]"]

@_reg("np_where")
def _gen_np_where(node, iv, ov):
    return ["import numpy as np"], [
        f"{ov['result']} = np.where({_val(node,iv,'condition')} > 0, {_val(node,iv,'x')}, {_val(node,iv,'y')})"
    ]

@_reg("np_dot")
def _gen_np_dot(node, iv, ov):
    return ["import numpy as np"], [f"{ov['result']} = np.dot({_val(node,iv,'a')}, {_val(node,iv,'b')})"]

@_reg("np_matmul")
def _gen_np_matmul(node, iv, ov):
    return ["import numpy as np"], [f"{ov['result']} = np.matmul({_val(node,iv,'a')}, {_val(node,iv,'b')})"]

@_reg("np_inv")
def _gen_np_inv(node, iv, ov):
    return ["import numpy as np"], [f"{ov['result']} = np.linalg.inv({_val(node,iv,'matrix')})"]

@_reg("np_eig")
def _gen_np_eig(node, iv, ov):
    return ["import numpy as np"], [
        f"{ov['eigenvalues']}, {ov['eigenvectors']} = np.linalg.eig({_val(node,iv,'matrix')})"
    ]

@_reg("np_svd")
def _gen_np_svd(node, iv, ov):
    return ["import numpy as np"], [
        f"{ov['U']}, {ov['S']}, {ov['Vt']} = np.linalg.svd({_val(node,iv,'matrix')}, full_matrices=False)"
    ]

@_reg("np_array_info")
def _gen_np_info(node, iv, ov):
    arr = _val(node,iv,'array')
    return ["import numpy as np"], [
        f"{ov['info']} = f\"shape={{list({arr}.shape)}} dtype={{{arr}.dtype}} "
        f"min={{{arr}.astype(float).min():.4g}} max={{{arr}.astype(float).max():.4g}} mean={{{arr}.astype(float).mean():.4g}}\""
    ]

@_reg("np_shape")
def _gen_np_shape(node, iv, ov):
    return [], [f"{ov['shape']} = str(list({_val(node,iv,'array')}.shape))"]


# ── Pandas ────────────────────────────────────────────────────────────────────

@_reg("pd_from_csv")
def _gen_pd_from_csv(node, iv, ov):
    return ["import pandas as pd"], [
        f"{ov['df']} = pd.read_csv({_val(node,iv,'path')}, sep={_val(node,iv,'sep')})"
    ]

@_reg("pd_from_numpy")
def _gen_pd_from_numpy(node, iv, ov):
    arr=_val(node,iv,'array'); cols=_val(node,iv,'columns')
    return ["import pandas as pd"], [
        f"_cols = [{cols}] if {cols}.strip() else None",
        f"{ov['df']} = pd.DataFrame({arr}, columns=_cols)",
    ]

@_reg("pd_from_dict")
def _gen_pd_from_dict(node, iv, ov):
    return ["import pandas as pd", "import json"], [
        f"{ov['df']} = pd.DataFrame(json.loads({_val(node,iv,'json_str')}))"
    ]

@_reg("pd_make_sample")
def _gen_pd_make_sample(node, iv, ov):
    rows=_val(node,iv,'rows'); cols=_val(node,iv,'cols'); seed=_val(node,iv,'seed')
    return ["import numpy as np", "import pandas as pd"], [
        f"_rng = np.random.default_rng({seed})",
        f"{ov['df']} = pd.DataFrame({{f'col_{{i}}': _rng.random({rows}) for i in range({cols})}}) ",
        f"{ov['df']}['label'] = _rng.integers(0, 2, {rows})",
    ]

@_reg("pd_head")
def _gen_pd_head(node, iv, ov):
    return [], [f"{ov['result']} = {_val(node,iv,'df')}.head({_val(node,iv,'n')}).to_string()"]

@_reg("pd_describe")
def _gen_pd_describe(node, iv, ov):
    return [], [f"{ov['result']} = {_val(node,iv,'df')}.describe().to_string()"]

@_reg("pd_info")
def _gen_pd_info(node, iv, ov):
    df = _val(node,iv,'df')
    return [], [f"{ov['result']} = f\"shape={{{df}.shape}} columns={{list({df}.columns)}}\""]

@_reg("pd_shape")
def _gen_pd_shape(node, iv, ov):
    df = _val(node,iv,'df')
    return [], [
        f"{ov['rows']} = {df}.shape[0]",
        f"{ov['cols']} = {df}.shape[1]",
    ]

@_reg("pd_select_cols")
def _gen_pd_select_cols(node, iv, ov):
    df=_val(node,iv,'df'); cols=_val(node,iv,'columns')
    return [], [
        f"_sel_cols = [c.strip() for c in {cols}.split(',') if c.strip()]",
        f"{ov['result']} = {df}[_sel_cols]",
    ]

@_reg("pd_drop_cols")
def _gen_pd_drop_cols(node, iv, ov):
    df=_val(node,iv,'df'); cols=_val(node,iv,'columns')
    return [], [
        f"_drop_cols = [c.strip() for c in {cols}.split(',') if c.strip()]",
        f"{ov['result']} = {df}.drop(columns=_drop_cols)",
    ]

@_reg("pd_filter_rows")
def _gen_pd_filter_rows(node, iv, ov):
    df=_val(node,iv,'df'); col=_val(node,iv,'column'); op=_val(node,iv,'op'); val=_val(node,iv,'value')
    return [], [f"{ov['result']} = {df}[{df}[{col}].apply(lambda _x: eval(f'{{_x}} {{{op}}} {{{val}}}'))]"]

@_reg("pd_get_column")
def _gen_pd_get_col(node, iv, ov):
    return [], [f"{ov['series']} = {_val(node,iv,'df')}[{_val(node,iv,'column')}]"]

@_reg("pd_dropna")
def _gen_pd_dropna(node, iv, ov):
    return [], [f"{ov['result']} = {_val(node,iv,'df')}.dropna()"]

@_reg("pd_fillna")
def _gen_pd_fillna(node, iv, ov):
    return [], [f"{ov['result']} = {_val(node,iv,'df')}.fillna({_val(node,iv,'value')})"]

@_reg("pd_sort")
def _gen_pd_sort(node, iv, ov):
    df=_val(node,iv,'df'); by=_val(node,iv,'by'); asc=_val(node,iv,'ascending')
    return [], [f"{ov['result']} = {df}.sort_values(by={by}, ascending={asc})"]

@_reg("pd_reset_index")
def _gen_pd_reset_index(node, iv, ov):
    return [], [f"{ov['result']} = {_val(node,iv,'df')}.reset_index(drop=True)"]

@_reg("pd_rename_col")
def _gen_pd_rename_col(node, iv, ov):
    df=_val(node,iv,'df'); old=_val(node,iv,'old_name'); new=_val(node,iv,'new_name')
    return [], [f"{ov['result']} = {df}.rename(columns={{{old}: {new}}})"]

@_reg("pd_to_numpy")
def _gen_pd_to_numpy(node, iv, ov):
    return [], [f"{ov['array']} = {_val(node,iv,'df')}.values"]

@_reg("pd_normalize")
def _gen_pd_normalize(node, iv, ov):
    df = _val(node,iv,'df')
    return [], [
        f"_num = {df}.select_dtypes(include='number')",
        f"_rng = _num.max() - _num.min(); _rng[_rng == 0] = 1",
        f"{ov['result']} = {df}.copy()",
        f"{ov['result']}[_num.columns] = (_num - _num.min()) / _rng",
    ]

@_reg("pd_groupby")
def _gen_pd_groupby(node, iv, ov):
    df=_val(node,iv,'df'); by=_val(node,iv,'by'); agg=_val(node,iv,'agg')
    return [], [f"{ov['result']} = {df}.groupby({by}).agg({agg}).reset_index()"]

@_reg("pd_correlation")
def _gen_pd_corr(node, iv, ov):
    return [], [f"{ov['result']} = {_val(node,iv,'df')}.select_dtypes(include='number').corr()"]

@_reg("pd_xy_split")
def _gen_pd_xy_split(node, iv, ov):
    df=_val(node,iv,'df'); lbl=_val(node,iv,'label_col')
    return [], [
        f"{ov['y']} = {df}[{lbl}]",
        f"{ov['X']} = {df}.drop(columns=[{lbl}])",
    ]

@_reg("pd_merge")
def _gen_pd_merge(node, iv, ov):
    l=_val(node,iv,'left'); r=_val(node,iv,'right'); on=_val(node,iv,'on'); how=_val(node,iv,'how')
    return ["import pandas as pd"], [
        f"{ov['result']} = pd.merge({l}, {r}, on={on}, how={how})"
    ]


# ── Sklearn ───────────────────────────────────────────────────────────────────

@_reg("sk_train_test_split")
def _gen_sk_tts(node, iv, ov):
    X=_val(node,iv,'X'); y=_val(node,iv,'y'); ts=_val(node,iv,'test_size'); rs=_val(node,iv,'random_state')
    return ["from sklearn.model_selection import train_test_split"], [
        f"{ov['X_train']}, {ov['X_test']}, {ov['y_train']}, {ov['y_test']} = "
        f"train_test_split({X}, {y}, test_size={ts}, random_state={rs})"
    ]

@_reg("sk_standard_scaler")
def _gen_sk_std_scaler(node, iv, ov):
    Xtr=_val(node,iv,'X_train'); Xte=_val(node,iv,'X_test')
    return ["from sklearn.preprocessing import StandardScaler"], [
        f"{ov['scaler']} = StandardScaler()",
        f"{ov['X_train_scaled']} = {ov['scaler']}.fit_transform({Xtr})",
        f"{ov['X_test_scaled']} = {ov['scaler']}.transform({Xte})",
    ]

@_reg("sk_minmax_scaler")
def _gen_sk_mm_scaler(node, iv, ov):
    Xtr=_val(node,iv,'X_train'); Xte=_val(node,iv,'X_test')
    return ["from sklearn.preprocessing import MinMaxScaler"], [
        f"{ov['scaler']} = MinMaxScaler()",
        f"{ov['X_train_scaled']} = {ov['scaler']}.fit_transform({Xtr})",
        f"{ov['X_test_scaled']} = {ov['scaler']}.transform({Xte})",
    ]

@_reg("sk_label_encoder")
def _gen_sk_le(node, iv, ov):
    return ["from sklearn.preprocessing import LabelEncoder"], [
        f"{ov['encoder']} = LabelEncoder()",
        f"{ov['encoded']} = {ov['encoder']}.fit_transform({_val(node,iv,'series')})",
    ]

@_reg("sk_onehot_encoder")
def _gen_sk_ohe(node, iv, ov):
    arr = _val(node,iv,'array')
    return ["import numpy as np", "from sklearn.preprocessing import OneHotEncoder"], [
        f"_ohe_arr = {arr}.reshape(-1,1) if {arr}.ndim == 1 else {arr}",
        f"{ov['encoder']} = OneHotEncoder(sparse_output=False)",
        f"{ov['encoded']} = {ov['encoder']}.fit_transform(_ohe_arr)",
    ]

@_reg("sk_linear_regression")
def _gen_sk_lr(node, iv, ov):
    return ["from sklearn.linear_model import LinearRegression"], [
        f"{ov['model']} = LinearRegression()",
        f"{ov['model']}.fit({_val(node,iv,'X_train')}, {_val(node,iv,'y_train')})",
    ]

@_reg("sk_logistic_regression")
def _gen_sk_logr(node, iv, ov):
    return ["from sklearn.linear_model import LogisticRegression"], [
        f"{ov['model']} = LogisticRegression(max_iter={_val(node,iv,'max_iter')})",
        f"{ov['model']}.fit({_val(node,iv,'X_train')}, {_val(node,iv,'y_train')})",
    ]

@_reg("sk_random_forest")
def _gen_sk_rf(node, iv, ov):
    n=_val(node,iv,'n_estimators'); rs=_val(node,iv,'random_state')
    return ["from sklearn.ensemble import RandomForestClassifier"], [
        f"{ov['model']} = RandomForestClassifier(n_estimators={n}, random_state={rs})",
        f"{ov['model']}.fit({_val(node,iv,'X_train')}, {_val(node,iv,'y_train')})",
    ]

@_reg("sk_svc")
def _gen_sk_svc(node, iv, ov):
    C=_val(node,iv,'C'); k=_val(node,iv,'kernel')
    return ["from sklearn.svm import SVC"], [
        f"{ov['model']} = SVC(C={C}, kernel={k})",
        f"{ov['model']}.fit({_val(node,iv,'X_train')}, {_val(node,iv,'y_train')})",
    ]

@_reg("sk_kmeans")
def _gen_sk_kmeans(node, iv, ov):
    k=_val(node,iv,'n_clusters'); rs=_val(node,iv,'random_state'); X=_val(node,iv,'X')
    return ["from sklearn.cluster import KMeans"], [
        f"{ov['model']} = KMeans(n_clusters={k}, random_state={rs}, n_init='auto')",
        f"{ov['labels']} = {ov['model']}.fit_predict({X})",
    ]

@_reg("sk_pca")
def _gen_sk_pca(node, iv, ov):
    n=_val(node,iv,'n_components'); X=_val(node,iv,'X')
    return ["from sklearn.decomposition import PCA"], [
        f"{ov['model']} = PCA(n_components={n})",
        f"{ov['transformed']} = {ov['model']}.fit_transform({X})",
    ]

@_reg("sk_gradient_boosting")
def _gen_sk_gb(node, iv, ov):
    n=_val(node,iv,'n_estimators'); lr=_val(node,iv,'learning_rate')
    return ["from sklearn.ensemble import GradientBoostingClassifier"], [
        f"{ov['model']} = GradientBoostingClassifier(n_estimators={n}, learning_rate={lr})",
        f"{ov['model']}.fit({_val(node,iv,'X_train')}, {_val(node,iv,'y_train')})",
    ]

@_reg("sk_predict")
def _gen_sk_predict(node, iv, ov):
    return [], [f"{ov['predictions']} = {_val(node,iv,'model')}.predict({_val(node,iv,'X')})"]

@_reg("sk_predict_proba")
def _gen_sk_predict_proba(node, iv, ov):
    return [], [f"{ov['probabilities']} = {_val(node,iv,'model')}.predict_proba({_val(node,iv,'X')})"]

@_reg("sk_accuracy")
def _gen_sk_accuracy(node, iv, ov):
    yt=_val(node,iv,'y_true'); yp=_val(node,iv,'y_pred')
    return [
        "from sklearn.metrics import accuracy_score, classification_report"
    ], [
        f"{ov['accuracy']} = float(accuracy_score({yt}, {yp}))",
        f"{ov['report']} = classification_report({yt}, {yp}, zero_division=0)",
    ]

@_reg("sk_confusion_matrix")
def _gen_sk_cm(node, iv, ov):
    yt=_val(node,iv,'y_true'); yp=_val(node,iv,'y_pred')
    return ["from sklearn.metrics import confusion_matrix"], [
        f"{ov['matrix']} = confusion_matrix({yt}, {yp})",
        f"{ov['info']} = 'Confusion matrix shape: ' + str({ov['matrix']}.shape)",
    ]

@_reg("sk_r2_score")
def _gen_sk_r2(node, iv, ov):
    yt=_val(node,iv,'y_true'); yp=_val(node,iv,'y_pred')
    return ["from sklearn.metrics import r2_score, mean_squared_error"], [
        f"{ov['r2']} = float(r2_score({yt}, {yp}))",
        f"{ov['mse']} = float(mean_squared_error({yt}, {yp}))",
    ]

@_reg("sk_cross_val_score")
def _gen_sk_cvs(node, iv, ov):
    m=_val(node,iv,'model'); X=_val(node,iv,'X'); y=_val(node,iv,'y'); cv=_val(node,iv,'cv')
    return ["from sklearn.model_selection import cross_val_score"], [
        f"{ov['scores']} = cross_val_score({m}, {X}, {y}, cv={cv})",
        f"{ov['mean']} = float({ov['scores']}.mean())",
    ]


# ── SciPy ─────────────────────────────────────────────────────────────────────

@_reg("sp_describe")
def _gen_sp_describe(node, iv, ov):
    arr = _val(node,iv,'array')
    return ["from scipy import stats"], [
        f"_spd = stats.describe({arr}.flatten())",
        f"{ov['info']} = f\"nobs={{_spd.nobs}} mean={{_spd.mean:.4g}} variance={{_spd.variance:.4g}} "
        f"skewness={{_spd.skewness:.4g}} kurtosis={{_spd.kurtosis:.4g}}\"",
    ]

@_reg("sp_norm_pdf")
def _gen_sp_norm_pdf(node, iv, ov):
    return ["from scipy.stats import norm"], [
        f"{ov['pdf']} = norm.pdf({_val(node,iv,'x')}, loc={_val(node,iv,'loc')}, scale={_val(node,iv,'scale')})"
    ]

@_reg("sp_ttest")
def _gen_sp_ttest(node, iv, ov):
    a=_val(node,iv,'a'); b=_val(node,iv,'b')
    return ["from scipy.stats import ttest_ind"], [
        f"_ttest = ttest_ind({a}.flatten(), {b}.flatten())",
        f"{ov['statistic']} = float(_ttest.statistic)",
        f"{ov['pvalue']} = float(_ttest.pvalue)",
    ]

@_reg("sp_pearsonr")
def _gen_sp_pearsonr(node, iv, ov):
    x=_val(node,iv,'x'); y=_val(node,iv,'y')
    return ["from scipy.stats import pearsonr"], [
        f"_pr = pearsonr({x}.flatten(), {y}.flatten())",
        f"{ov['r']} = float(_pr.statistic)",
        f"{ov['pvalue']} = float(_pr.pvalue)",
    ]

@_reg("sp_histogram")
def _gen_sp_hist(node, iv, ov):
    return ["import numpy as np"], [
        f"{ov['counts']}, {ov['bin_edges']} = np.histogram({_val(node,iv,'array')}.flatten(), bins={_val(node,iv,'bins')})"
    ]

@_reg("sp_fft")
def _gen_sp_fft(node, iv, ov):
    sig = _val(node,iv,'signal')
    return ["import numpy as np"], [
        f"_fft_sig = {sig}.flatten()",
        f"_fft_vals = np.fft.rfft(_fft_sig)",
        f"{ov['magnitude']} = np.abs(_fft_vals)",
        f"{ov['frequencies']} = np.fft.rfftfreq(len(_fft_sig))",
    ]

@_reg("sp_butterworth")
def _gen_sp_butter(node, iv, ov):
    sig=_val(node,iv,'signal'); cut=_val(node,iv,'cutoff'); ord=_val(node,iv,'order'); bt=_val(node,iv,'btype')
    return ["from scipy.signal import butter, sosfilt"], [
        f"_sos = butter(int({ord}), float({cut}), btype={bt}, output='sos')",
        f"{ov['filtered']} = sosfilt(_sos, {sig}.flatten())",
    ]

@_reg("sp_convolve")
def _gen_sp_convolve(node, iv, ov):
    a=_val(node,iv,'a'); b=_val(node,iv,'b'); mode=_val(node,iv,'mode')
    return ["import numpy as np"], [
        f"{ov['result']} = np.convolve({a}.flatten(), {b}.flatten(), mode={mode})"
    ]

@_reg("sp_interp1d")
def _gen_sp_interp(node, iv, ov):
    x=_val(node,iv,'x'); y=_val(node,iv,'y'); xn=_val(node,iv,'x_new'); k=_val(node,iv,'kind')
    return ["from scipy.interpolate import interp1d"], [
        f"_f_interp = interp1d({x}.flatten(), {y}.flatten(), kind={k}, bounds_error=False, fill_value='extrapolate')",
        f"{ov['y_new']} = _f_interp({xn}.flatten())",
    ]

@_reg("sp_curve_fit")
def _gen_sp_curve_fit(node, iv, ov):
    x=_val(node,iv,'x'); y=_val(node,iv,'y'); fs=_val(node,iv,'func_str')
    return ["import numpy as np", "from scipy.optimize import curve_fit"], [
        f"_param_names = [p for p in ['a','b','c','d'] if p in {fs}]",
        f"_fn = eval(f\"lambda x,{{','.join(_param_names)}}: {{{fs}}}\")",
        f"{ov['params']}, _ = curve_fit(_fn, {x}.flatten(), {y}.flatten(), maxfev=5000)",
        f"{ov['info']} = ', '.join(f'{{n}}={{v:.6g}}' for n,v in zip(_param_names, {ov['params']}))",
    ]


# ── PyTorch — layers ──────────────────────────────────────────────────────────
# Each layer node has tensor_in / tensor_out. The graph topology defines the
# model. For code export, each node generates its forward pass inline.

def _act_expr(act_name: str) -> str | None:
    key = (act_name or "").strip().lower().replace(" ", "").replace("_", "")
    return {
        "relu":      "nn.ReLU()",
        "leakyrelu": "nn.LeakyReLU(0.01)",
        "sigmoid":   "nn.Sigmoid()",
        "tanh":      "nn.Tanh()",
        "gelu":      "nn.GELU()",
        "elu":       "nn.ELU()",
        "silu":      "nn.SiLU()",
        "swish":     "nn.SiLU()",
        "softmax":   "nn.Softmax(dim=1)",
    }.get(key)


def _layer_fwd(ov, iv, layer_var: str, layer_expr: str, act_name: str = "") -> list[str]:
    """Generate: instantiate layer, apply to tensor_in, store in tensor_out."""
    tin  = iv.get("tensor_in") or "_x"
    tout = ov.get("tensor_out", "_out")
    act  = _act_expr(act_name)
    lines = [f"{layer_var} = {layer_expr}"]
    if act:
        lines += [
            f"with torch.no_grad():",
            f"    _tmp = {layer_var}({tin})",
            f"    {tout} = {act}(_tmp)",
        ]
    else:
        lines.append(f"{tout} = {layer_var}({tin})")
    return lines


@_reg("pt_linear")
def _gen_pt_linear(node, iv, ov):
    lv  = f"_lin_{node.id[:6]}"
    act = node.inputs["activation"].default_value if "activation" in node.inputs else ""
    layer = (f"nn.Linear({_val(node,iv,'in_features')}, {_val(node,iv,'out_features')}, "
             f"bias={_val(node,iv,'bias')})")
    return ["import torch", "import torch.nn as nn"], _layer_fwd(ov, iv, lv, layer, act)

@_reg("pt_conv2d")
def _gen_pt_conv2d(node, iv, ov):
    lv  = f"_conv_{node.id[:6]}"
    act = node.inputs["activation"].default_value if "activation" in node.inputs else ""
    layer = (f"nn.Conv2d({_val(node,iv,'in_ch')}, {_val(node,iv,'out_ch')}, "
             f"{_val(node,iv,'kernel')}, stride={_val(node,iv,'stride')}, "
             f"padding={_val(node,iv,'padding')})")
    return ["import torch", "import torch.nn as nn"], _layer_fwd(ov, iv, lv, layer, act)

@_reg("pt_batchnorm1d")
def _gen_pt_bn1d(node, iv, ov):
    lv = f"_bn1_{node.id[:6]}"
    return ["import torch", "import torch.nn as nn"], _layer_fwd(
        ov, iv, lv, f"nn.BatchNorm1d({_val(node,iv,'num_features')})")

@_reg("pt_batchnorm2d")
def _gen_pt_bn2d(node, iv, ov):
    lv = f"_bn2_{node.id[:6]}"
    return ["import torch", "import torch.nn as nn"], _layer_fwd(
        ov, iv, lv, f"nn.BatchNorm2d({_val(node,iv,'num_features')})")

@_reg("pt_dropout")
def _gen_pt_dropout(node, iv, ov):
    lv = f"_drop_{node.id[:6]}"
    return ["import torch", "import torch.nn as nn"], _layer_fwd(
        ov, iv, lv, f"nn.Dropout({_val(node,iv,'p')})")

@_reg("pt_flatten")
def _gen_pt_flatten(node, iv, ov):
    lv = f"_flat_{node.id[:6]}"
    return ["import torch", "import torch.nn as nn"], _layer_fwd(
        ov, iv, lv, f"nn.Flatten(start_dim={_val(node,iv,'start_dim')})")

@_reg("pt_embedding")
def _gen_pt_embedding(node, iv, ov):
    lv = f"_emb_{node.id[:6]}"
    return ["import torch", "import torch.nn as nn"], _layer_fwd(
        ov, iv, lv, f"nn.Embedding({_val(node,iv,'num_embeddings')}, {_val(node,iv,'embedding_dim')})")

@_reg("pt_maxpool2d")
def _gen_pt_maxpool(node, iv, ov):
    lv = f"_mxp_{node.id[:6]}"
    return ["import torch", "import torch.nn as nn"], _layer_fwd(
        ov, iv, lv, f"nn.MaxPool2d({_val(node,iv,'kernel')}, stride={_val(node,iv,'stride')})")

@_reg("pt_avgpool2d")
def _gen_pt_avgpool(node, iv, ov):
    lv = f"_avp_{node.id[:6]}"
    return ["import torch", "import torch.nn as nn"], _layer_fwd(
        ov, iv, lv, f"nn.AvgPool2d({_val(node,iv,'kernel')}, stride={_val(node,iv,'stride')})")

@_reg("pt_activation")
def _gen_pt_activation(node, iv, ov):
    act_name = node.inputs["activation"].default_value if "activation" in node.inputs else "relu"
    act_expr = _act_expr(act_name) or "nn.ReLU()"
    tin  = iv.get("tensor_in") or "_x"
    tout = ov.get("tensor_out", "_out")
    return ["import torch.nn as nn"], [
        f"_act_{node.id[:6]} = {act_expr}",
        f"{tout} = _act_{node.id[:6]}({tin})",
    ]

@_reg("pt_sequential")
def _gen_pt_sequential(node, iv, ov):
    layers = [iv.get(f"layer_{i}") for i in range(1, 9) if iv.get(f"layer_{i}") is not None]
    if not layers:
        return ["import torch.nn as nn"], [f"{ov['model']} = nn.Sequential()  # no layers connected"]
    return ["import torch.nn as nn"], [f"{ov['model']} = nn.Sequential({', '.join(layers)})"]


# ── PyTorch — tensor ops ──────────────────────────────────────────────────────

@_reg("pt_rand_tensor")
def _gen_pt_rand(node, iv, ov):
    shape=_val(node,iv,'shape'); rg=_val(node,iv,'requires_grad')
    return ["import torch"], [
        f"_shape = tuple(int(x) for x in {shape}.split(',') if x.strip())",
        f"{ov['tensor']} = torch.randn(_shape, requires_grad={rg})",
    ]

@_reg("pt_zeros_tensor")
def _gen_pt_zeros(node, iv, ov):
    shape=_val(node,iv,'shape')
    return ["import torch"], [
        f"_shape = tuple(int(x) for x in {shape}.split(',') if x.strip())",
        f"{ov['tensor']} = torch.zeros(_shape)",
    ]

@_reg("pt_ones_tensor")
def _gen_pt_ones(node, iv, ov):
    shape=_val(node,iv,'shape')
    return ["import torch"], [
        f"_shape = tuple(int(x) for x in {shape}.split(',') if x.strip())",
        f"{ov['tensor']} = torch.ones(_shape)",
    ]

@_reg("pt_tensor_from_list")
def _gen_pt_tensor_list(node, iv, ov):
    vals=_val(node,iv,'values')
    return ["import torch"], [
        f"{ov['tensor']} = torch.tensor([float(x) for x in {vals}.split(',') if x.strip()])"
    ]

@_reg("pt_forward_pass")
def _gen_pt_forward(node, iv, ov):
    m=_val(node,iv,'model'); inp=_val(node,iv,'input')
    return ["import torch"], [
        f"{m}.eval()",
        f"with torch.no_grad():",
        f"    {ov['output']} = {m}({inp})",
    ]

@_reg("pt_tensor_shape")
def _gen_pt_tensor_shape(node, iv, ov):
    return [], [f"{ov['shape']} = str(list({_val(node,iv,'tensor')}.shape))"]

@_reg("pt_tensor_info")
def _gen_pt_tensor_info(node, iv, ov):
    t=_val(node,iv,'tensor')
    return [], [
        f"{ov['info']} = f\"shape={{list({t}.shape)}} dtype={{{t}.dtype}} min={{{t}.min():.4f}} max={{{t}.max():.4f}}\""
    ]

@_reg("pt_tensor_add")
def _gen_pt_tadd(node, iv, ov):
    return [], [f"{ov['result']} = {_val(node,iv,'a')} + {_val(node,iv,'b')}"]

@_reg("pt_tensor_mul")
def _gen_pt_tmul(node, iv, ov):
    return [], [f"{ov['result']} = {_val(node,iv,'a')} * {_val(node,iv,'b')}"]

@_reg("pt_argmax")
def _gen_pt_argmax(node, iv, ov):
    return ["import torch"], [
        f"{ov['result']} = torch.argmax({_val(node,iv,'tensor')}, dim={_val(node,iv,'dim')})"
    ]

@_reg("pt_softmax_op")
def _gen_pt_softmax_op(node, iv, ov):
    return ["import torch"], [
        f"{ov['result']} = torch.softmax({_val(node,iv,'tensor')}, dim={_val(node,iv,'dim')})"
    ]

@_reg("pt_print_tensor")
def _gen_pt_print_tensor(node, iv, ov):
    t=_val(node,iv,'tensor'); lbl=_val(node,iv,'label')
    return [], [
        f"{ov['passthrough']} = {t}",
        f"print(f\"{{{lbl}}}: shape={{list({t}.shape)}}\" if {lbl} else f\"shape={{list({t}.shape)}}\")",
    ]


# ── PyTorch — losses ──────────────────────────────────────────────────────────

@_reg("pt_mse_loss")
def _gen_pt_mse(node, iv, ov):
    return ["import torch.nn as nn"], [f"{ov['loss_fn']} = nn.MSELoss(reduction={_val(node,iv,'reduction')})"]

@_reg("pt_cross_entropy")
def _gen_pt_ce(node, iv, ov):
    return ["import torch.nn as nn"], [f"{ov['loss_fn']} = nn.CrossEntropyLoss()"]

@_reg("pt_bce_loss")
def _gen_pt_bce(node, iv, ov):
    return ["import torch.nn as nn"], [f"{ov['loss_fn']} = nn.BCELoss(reduction={_val(node,iv,'reduction')})"]

@_reg("pt_bce_logits")
def _gen_pt_bce_logits(node, iv, ov):
    return ["import torch.nn as nn"], [f"{ov['loss_fn']} = nn.BCEWithLogitsLoss(reduction={_val(node,iv,'reduction')})"]

@_reg("pt_l1_loss")
def _gen_pt_l1(node, iv, ov):
    return ["import torch.nn as nn"], [f"{ov['loss_fn']} = nn.L1Loss(reduction={_val(node,iv,'reduction')})"]


# ── PyTorch — optimizers ──────────────────────────────────────────────────────

@_reg("pt_adam")
def _gen_pt_adam(node, iv, ov):
    m=_val(node,iv,'model'); lr=_val(node,iv,'lr'); wd=_val(node,iv,'weight_decay')
    return ["import torch.optim as optim"], [
        f"{ov['optimizer']} = optim.Adam({m}.parameters(), lr={lr}, weight_decay={wd})"
    ]

@_reg("pt_sgd")
def _gen_pt_sgd(node, iv, ov):
    m=_val(node,iv,'model'); lr=_val(node,iv,'lr'); mom=_val(node,iv,'momentum'); wd=_val(node,iv,'weight_decay')
    return ["import torch.optim as optim"], [
        f"{ov['optimizer']} = optim.SGD({m}.parameters(), lr={lr}, momentum={mom}, weight_decay={wd})"
    ]

@_reg("pt_adamw")
def _gen_pt_adamw(node, iv, ov):
    m=_val(node,iv,'model'); lr=_val(node,iv,'lr'); wd=_val(node,iv,'weight_decay')
    return ["import torch.optim as optim"], [
        f"{ov['optimizer']} = optim.AdamW({m}.parameters(), lr={lr}, weight_decay={wd})"
    ]


# ── PyTorch — LR schedulers ──────────────────────────────────────────────────

@_reg("pt_step_lr")
def _gen_pt_step_lr(node, iv, ov):
    opt=_val(node,iv,'optimizer'); ss=_val(node,iv,'step_size'); g=_val(node,iv,'gamma')
    return ["from torch.optim.lr_scheduler import StepLR"], [
        f"{ov['scheduler']} = StepLR({opt}, step_size={ss}, gamma={g})"
    ]

@_reg("pt_multistep_lr")
def _gen_pt_multistep_lr(node, iv, ov):
    opt=_val(node,iv,'optimizer'); ms=_val(node,iv,'milestones'); g=_val(node,iv,'gamma')
    return ["from torch.optim.lr_scheduler import MultiStepLR"], [
        f"{ov['scheduler']} = MultiStepLR({opt}, milestones={ms}, gamma={g})"
    ]

@_reg("pt_exponential_lr")
def _gen_pt_exp_lr(node, iv, ov):
    opt=_val(node,iv,'optimizer'); g=_val(node,iv,'gamma')
    return ["from torch.optim.lr_scheduler import ExponentialLR"], [
        f"{ov['scheduler']} = ExponentialLR({opt}, gamma={g})"
    ]

@_reg("pt_cosine_lr")
def _gen_pt_cosine_lr(node, iv, ov):
    opt=_val(node,iv,'optimizer'); t=_val(node,iv,'T_max'); eta=_val(node,iv,'eta_min')
    return ["from torch.optim.lr_scheduler import CosineAnnealingLR"], [
        f"{ov['scheduler']} = CosineAnnealingLR({opt}, T_max={t}, eta_min={eta})"
    ]

@_reg("pt_reduce_lr_plateau")
def _gen_pt_reduce_lr(node, iv, ov):
    opt=_val(node,iv,'optimizer'); mode=_val(node,iv,'mode')
    factor=_val(node,iv,'factor'); patience=_val(node,iv,'patience')
    return ["from torch.optim.lr_scheduler import ReduceLROnPlateau"], [
        f"{ov['scheduler']} = ReduceLROnPlateau({opt}, mode={mode}, factor={factor}, patience={patience})"
    ]


# ── PyTorch — data ────────────────────────────────────────────────────────────

@_reg("pt_mnist")
def _gen_pt_mnist(node, iv, ov):
    bs=_val(node,iv,'batch_size'); train=_val(node,iv,'train')
    dl=_val(node,iv,'download'); shuffle=_val(node,iv,'shuffle')
    return [
        "import torch",
        "from torchvision import datasets, transforms",
        "from torch.utils.data import DataLoader",
    ], [
        f"_mnist = datasets.MNIST(root='./data', train={train}, download={dl}, transform=transforms.ToTensor())",
        f"{ov['dataloader']} = DataLoader(_mnist, batch_size={bs}, shuffle={shuffle})",
    ]

@_reg("pt_cifar10")
def _gen_pt_cifar(node, iv, ov):
    bs=_val(node,iv,'batch_size'); train=_val(node,iv,'train')
    dl=_val(node,iv,'download'); shuffle=_val(node,iv,'shuffle')
    return [
        "import torch",
        "from torchvision import datasets, transforms",
        "from torch.utils.data import DataLoader",
    ], [
        f"_cifar = datasets.CIFAR10(root='./data', train={train}, download={dl}, transform=transforms.ToTensor())",
        f"{ov['dataloader']} = DataLoader(_cifar, batch_size={bs}, shuffle={shuffle})",
    ]

@_reg("pt_dataloader_info")
def _gen_pt_dl_info(node, iv, ov):
    dl = _val(node,iv,'dataloader')
    return [], [f"{ov['info']} = f\"batches={{len({dl})}} batch_size={{{dl}.batch_size}}\""]


# ── PyTorch — training config (crown jewel) ───────────────────────────────────

@_reg("pt_training_config")
def _gen_pt_training_config(node, iv, ov):
    """Expand into a full training loop with built-in optimizer/loss/scheduler."""
    model      = iv.get('tensor_in') or '_model'  # last tensor var = predictions var
    loader     = _val(node, iv, 'dataloader')
    val_loader = iv.get('val_dataloader')
    epochs     = _val(node, iv, 'epochs')
    device     = _val(node, iv, 'device')

    # Optimizer settings
    opt_name  = node.inputs["optimizer"].default_value if "optimizer" in node.inputs else "adam"
    lr        = node.inputs["lr"].default_value        if "lr"        in node.inputs else 0.001
    wd        = node.inputs["weight_decay"].default_value if "weight_decay" in node.inputs else 0.0
    mom       = node.inputs["momentum"].default_value  if "momentum"  in node.inputs else 0.9
    loss_name = node.inputs["loss"].default_value      if "loss"      in node.inputs else "crossentropy"
    sch_name  = node.inputs["scheduler"].default_value if "scheduler" in node.inputs else "none"
    step_size = node.inputs["step_size"].default_value if "step_size" in node.inputs else 10
    gamma     = node.inputs["gamma"].default_value     if "gamma"     in node.inputs else 0.1
    T_max     = node.inputs["T_max"].default_value     if "T_max"     in node.inputs else 50

    # Map string -> constructor expression
    opt_map = {
        "adamw":   f"torch.optim.AdamW({model}.parameters(), lr={lr}, weight_decay={wd})",
        "sgd":     f"torch.optim.SGD({model}.parameters(), lr={lr}, weight_decay={wd}, momentum={mom})",
        "rmsprop": f"torch.optim.RMSprop({model}.parameters(), lr={lr}, weight_decay={wd})",
    }
    opt_expr  = opt_map.get(str(opt_name).strip().lower(), f"torch.optim.Adam({model}.parameters(), lr={lr}, weight_decay={wd})")

    loss_map = {
        "mse":           "torch.nn.MSELoss()",
        "bce":           "torch.nn.BCELoss()",
        "bcewithlogits": "torch.nn.BCEWithLogitsLoss()",
        "l1":            "torch.nn.L1Loss()",
    }
    loss_expr = loss_map.get(str(loss_name).strip().lower().replace("-",""), "torch.nn.CrossEntropyLoss()")

    sch_key = str(sch_name).strip().lower().replace("_","")
    if sch_key == "steplr":
        sch_expr = f"torch.optim.lr_scheduler.StepLR(_optimizer, step_size={step_size}, gamma={gamma})"
    elif sch_key == "cosine":
        sch_expr = f"torch.optim.lr_scheduler.CosineAnnealingLR(_optimizer, T_max={T_max})"
    elif sch_key == "reducelr":
        sch_expr = f"torch.optim.lr_scheduler.ReduceLROnPlateau(_optimizer, factor={gamma}, patience={step_size})"
    else:
        sch_expr = None

    lines = [
        "# Training setup",
        f"_device   = torch.device({device})",
        f"{model}   = {model}.to(_device)",
        f"_optimizer = {opt_expr}",
        f"_loss_fn   = {loss_expr}",
    ]
    if sch_expr:
        lines.append(f"_scheduler = {sch_expr}")

    lines += [
        f"# Training loop",
        f"for _epoch in range(1, int({epochs}) + 1):",
        f"    {model}.train()",
        f"    _train_loss = 0.0",
        f"    for _batch in {loader}:",
        f"        if isinstance(_batch, (list, tuple)):",
        f"            _x, _y = _batch[0].to(_device), _batch[1].to(_device)",
        f"        else:",
        f"            _x = _batch.to(_device); _y = _x",
        f"        _optimizer.zero_grad()",
        f"        _out = {model}(_x)",
        f"        _loss = _loss_fn(_out, _y)",
        f"        _loss.backward()",
        f"        _optimizer.step()",
        f"        _train_loss += _loss.item()",
        f"    _train_loss /= len({loader})",
    ]

    if val_loader is not None:
        lines += [
            f"    # Validation",
            f"    {model}.eval()",
            f"    _val_loss = 0.0",
            f"    with torch.no_grad():",
            f"        for _vbatch in {val_loader}:",
            f"            if isinstance(_vbatch, (list, tuple)):",
            f"                _vx, _vy = _vbatch[0].to(_device), _vbatch[1].to(_device)",
            f"            else:",
            f"                _vx = _vbatch.to(_device); _vy = _vx",
            f"            _val_loss += _loss_fn({model}(_vx), _vy).item()",
            f"    _val_loss /= len({val_loader})",
        ]

    if sch_expr:
        lines += [
            f"    # LR scheduler step",
            f"    from torch.optim.lr_scheduler import ReduceLROnPlateau as _RLRP",
            f"    if isinstance(_scheduler, _RLRP):",
            f"        _scheduler.step({'_val_loss' if val_loader is not None else '_train_loss'})",
            f"    else:",
            f"        _scheduler.step()",
        ]

    if val_loader is not None:
        lines.append(
            f"    print(f\"Epoch {{_epoch}}/{{{epochs}}}  train={{_train_loss:.6f}}  val={{_val_loss:.6f}}\")"
        )
    else:
        lines.append(
            f"    print(f\"Epoch {{_epoch}}/{{{epochs}}}  loss={{_train_loss:.6f}}\")"
        )

    return ["import torch"], lines


@_reg("pt_inference_model")
def _gen_pt_inference_model(node, iv, ov):
    m = _val(node,iv,'model')
    return [], [f"{ov['model']} = {m}  # inference model marker"]

@_reg("pt_forward_pass")
def _gen_pt_fwd2(node, iv, ov):
    m=_val(node,iv,'model'); inp=_val(node,iv,'input')
    return ["import torch"], [
        f"{m}.eval()",
        f"with torch.no_grad():",
        f"    {ov['output']} = {m}({inp})",
    ]


# ── PyTorch — dataset sources ─────────────────────────────────────────────────

@_reg("pt_csv_dataset")
def _gen_pt_csv_dataset(node, iv, ov):
    path=_val(node,iv,'file_path'); target=_val(node,iv,'target_col')
    feat=_val(node,iv,'feature_cols'); norm=_val(node,iv,'normalize')
    dsv = ov['dataset']; infov = ov['info']
    return ["import pandas as pd","import torch","from torch.utils.data import TensorDataset"], [
        f"_df_{dsv} = pd.read_csv({path})",
        f"_feat_cols = [c for c in _df_{dsv}.columns if c != {target}] if not {feat} else [{feat}]",
        f"_X = _df_{dsv}[_feat_cols].values.astype('float32')",
        f"_y = _df_{dsv}[{target}].values",
        f"{dsv} = TensorDataset(torch.tensor(_X), torch.tensor(_y.astype('int64')))",
        f"{infov} = f'CSVDataset: {{len({dsv})}} samples'",
    ]

@_reg("pt_numpy_dataset")
def _gen_pt_numpy_dataset(node, iv, ov):
    X=_val(node,iv,'X'); y=_val(node,iv,'y')
    dsv = ov['dataset']; infov = ov['info']
    return ["import torch","from torch.utils.data import TensorDataset"], [
        f"{dsv} = TensorDataset(torch.tensor({X}.astype('float32')), torch.tensor({y}.astype('int64')))",
        f"{infov} = f'NumpyDataset: {{len({dsv})}} samples'",
    ]

@_reg("pt_image_folder_dataset")
def _gen_pt_image_folder(node, iv, ov):
    root=_val(node,iv,'root_path'); tf=iv.get('transform')
    tf_arg = f", transform={tf}" if tf else ""
    dsv = ov['dataset']; infov = ov['info']; cnv = ov['class_names']
    return ["from torchvision.datasets import ImageFolder"], [
        f"{dsv} = ImageFolder(root={root}{tf_arg})",
        f"{cnv} = ', '.join({dsv}.classes)",
        f"{infov} = f'ImageFolder: {{len({dsv})}} images'",
    ]

@_reg("pt_hf_dataset")
def _gen_pt_hf_dataset(node, iv, ov):
    name=_val(node,iv,'dataset_name'); split=_val(node,iv,'split')
    dsv = ov['dataset']; infov = ov['info']
    return ["from datasets import load_dataset"], [
        f"{dsv} = load_dataset({name}, split={split})",
        f"{infov} = f'HFDataset: {{len({dsv})}} samples'",
    ]

for _t in ("pt_audio_folder_dataset",):
    _GENERATORS[_t] = lambda node, iv, ov: ([], [f"# [{node.label}]: load dataset manually — audio loading requires runtime context"])


# ── PyTorch — transforms ──────────────────────────────────────────────────────

@_reg("pt_compose_transforms")
def _gen_pt_compose(node, iv, ov):
    ts = [iv.get(f"t{i}") for i in range(1,7) if iv.get(f"t{i}")]
    parts = ", ".join(ts)
    return ["from torchvision.transforms import Compose"], [
        f"{ov['transform']} = Compose([{parts}])"
    ]

@_reg("pt_to_tensor_transform")
def _gen_pt_to_tensor(node, iv, ov):
    return ["from torchvision.transforms import ToTensor"], [f"{ov['transform']} = ToTensor()"]

@_reg("pt_resize_transform")
def _gen_pt_resize(node, iv, ov):
    h=_val(node,iv,'height'); w=_val(node,iv,'width')
    return ["from torchvision.transforms import Resize"], [f"{ov['transform']} = Resize(({h}, {w}))"]

@_reg("pt_normalize_transform")
def _gen_pt_normalize(node, iv, ov):
    m=_val(node,iv,'mean'); s=_val(node,iv,'std')
    return ["from torchvision.transforms import Normalize"], [
        f"{ov['transform']} = Normalize(mean=[float(x) for x in {m}.split(',')], std=[float(x) for x in {s}.split(',')])"
    ]

@_reg("pt_random_hflip_transform")
def _gen_pt_hflip(node, iv, ov):
    p=_val(node,iv,'p')
    return ["from torchvision.transforms import RandomHorizontalFlip"], [f"{ov['transform']} = RandomHorizontalFlip(p={p})"]

@_reg("pt_random_vflip_transform")
def _gen_pt_vflip(node, iv, ov):
    p=_val(node,iv,'p')
    return ["from torchvision.transforms import RandomVerticalFlip"], [f"{ov['transform']} = RandomVerticalFlip(p={p})"]

@_reg("pt_center_crop_transform")
def _gen_pt_center_crop(node, iv, ov):
    s=_val(node,iv,'size')
    return ["from torchvision.transforms import CenterCrop"], [f"{ov['transform']} = CenterCrop({s})"]

@_reg("pt_random_crop_transform")
def _gen_pt_random_crop(node, iv, ov):
    s=_val(node,iv,'size'); p=_val(node,iv,'padding')
    return ["from torchvision.transforms import RandomCrop"], [f"{ov['transform']} = RandomCrop({s}, padding={p})"]

@_reg("pt_grayscale_transform")
def _gen_pt_grayscale(node, iv, ov):
    c=_val(node,iv,'num_channels')
    return ["from torchvision.transforms import Grayscale"], [f"{ov['transform']} = Grayscale(num_output_channels={c})"]

@_reg("pt_color_jitter_transform")
def _gen_pt_color_jitter(node, iv, ov):
    b=_val(node,iv,'brightness'); c=_val(node,iv,'contrast'); s=_val(node,iv,'saturation'); h=_val(node,iv,'hue')
    return ["from torchvision.transforms import ColorJitter"], [
        f"{ov['transform']} = ColorJitter(brightness={b}, contrast={c}, saturation={s}, hue={h})"
    ]

@_reg("pt_mel_spectrogram_transform")
def _gen_pt_mel_spec(node, iv, ov):
    sr=_val(node,iv,'sample_rate'); nm=_val(node,iv,'n_mels'); nf=_val(node,iv,'n_fft')
    return ["import torchaudio.transforms as T"], [
        f"{ov['transform']} = T.MelSpectrogram(sample_rate={sr}, n_mels={nm}, n_fft={nf})"
    ]

@_reg("pt_hf_tokenizer_transform")
def _gen_pt_hf_tok(node, iv, ov):
    name=_val(node,iv,'model_name'); ml=_val(node,iv,'max_length')
    tfv = ov['transform']
    return ["from transformers import AutoTokenizer"], [
        f"_tok_{tfv} = AutoTokenizer.from_pretrained({name})",
        f"{tfv} = lambda text: _tok_{tfv}(text, max_length={ml}, padding='max_length', truncation=True, return_tensors='pt')",
    ]


# ── PyTorch — dataset ops ─────────────────────────────────────────────────────

@_reg("pt_apply_transform")
def _gen_pt_apply_transform(node, iv, ov):
    ds=_val(node,iv,'dataset'); tf=_val(node,iv,'transform')
    dsv = ov['dataset']
    return ["from torch.utils.data import Dataset as _TDS"], [
        f"class _TD_{dsv}(_TDS):",
        f"    def __init__(self,b,t): self.b=b; self.t=t",
        f"    def __len__(self): return len(self.b)",
        f"    def __getitem__(self,i):",
        f"        s=self.b[i]; return (self.t(s[0]),)+tuple(s[1:]) if isinstance(s,(list,tuple)) else self.t(s)",
        f"{dsv} = _TD_{dsv}({ds}, {tf})",
    ]

@_reg("pt_train_val_split")
def _gen_pt_tv_split(node, iv, ov):
    ds=_val(node,iv,'dataset'); vr=_val(node,iv,'val_ratio'); seed=_val(node,iv,'seed')
    return ["import torch","from torch.utils.data import random_split"], [
        f"_n_{ds} = len({ds})",
        f"_nv_{ds} = max(1, int(_n_{ds} * {vr}))",
        f"_nt_{ds} = _n_{ds} - _nv_{ds}",
        f"{ov['train_dataset']}, {ov['val_dataset']} = random_split({ds}, [_nt_{ds}, _nv_{ds}], generator=torch.Generator().manual_seed({seed}))",
    ]

@_reg("pt_train_val_test_split")
def _gen_pt_tvt_split(node, iv, ov):
    ds=_val(node,iv,'dataset'); vr=_val(node,iv,'val_ratio'); tr=_val(node,iv,'test_ratio'); seed=_val(node,iv,'seed')
    return ["import torch","from torch.utils.data import random_split"], [
        f"_n_{ds} = len({ds})",
        f"_nv_{ds} = max(1, int(_n_{ds} * {vr}))",
        f"_nte_{ds} = max(1, int(_n_{ds} * {tr}))",
        f"_ntr_{ds} = _n_{ds} - _nv_{ds} - _nte_{ds}",
        f"{ov['train_dataset']}, {ov['val_dataset']}, {ov['test_dataset']} = random_split({ds}, [_ntr_{ds}, _nv_{ds}, _nte_{ds}], generator=torch.Generator().manual_seed({seed}))",
    ]

@_reg("pt_dataloader")
def _gen_pt_dataloader(node, iv, ov):
    ds=_val(node,iv,'dataset'); bs=_val(node,iv,'batch_size'); sh=_val(node,iv,'shuffle')
    nw=_val(node,iv,'num_workers'); pm=_val(node,iv,'pin_memory'); dl=_val(node,iv,'drop_last')
    dlv = ov['dataloader']; infov = ov['info']
    return ["from torch.utils.data import DataLoader"], [
        f"{dlv} = DataLoader({ds}, batch_size={bs}, shuffle={sh}, num_workers={nw}, pin_memory={pm}, drop_last={dl})",
        f"{infov} = f'DataLoader: {{len({dlv})}} batches'",
    ]

@_reg("pt_dataset_info")
def _gen_pt_dataset_info(node, iv, ov):
    ds=_val(node,iv,'dataset')
    return [], [f"{ov['info']} = f'Dataset: {{len({ds})}} samples'"]


# ── PyTorch — persistence ─────────────────────────────────────────────────────

@_reg("pt_save_weights")
def _gen_pt_save_weights(node, iv, ov):
    model = iv.get("model", "None")
    path  = iv.get("path",  repr(node.inputs["path"].default_value))
    lines = [
        f"if {model} is not None:",
        f"    torch.save({model}.state_dict(), {path})",
    ]
    passthrough = []
    if "model" in ov:
        passthrough.append(f"{ov['model']} = {model}")
    if "path" in ov:
        passthrough.append(f"{ov['path']} = {path}")
    return ["import torch"], lines + passthrough


@_reg("pt_load_weights")
def _gen_pt_load_weights(node, iv, ov):
    model  = iv.get("model",  "None")
    path   = iv.get("path",   repr(node.inputs["path"].default_value))
    device = iv.get("device", repr(node.inputs["device"].default_value))
    out    = ov.get("model", "_model")
    lines  = [
        f"_ckpt_state = torch.load({path}, map_location={device}, weights_only=True)",
        f"{model}.load_state_dict(_ckpt_state)",
        f"{model}.to({device})",
        f"{out} = {model}",
    ]
    return ["import torch"], lines


@_reg("pt_save_checkpoint")
def _gen_pt_save_checkpoint(node, iv, ov):
    model     = iv.get("model",     "None")
    optimizer = iv.get("optimizer", "None")
    epoch     = iv.get("epoch",     "0")
    loss      = iv.get("loss",      "0.0")
    path      = iv.get("path",      repr(node.inputs["path"].default_value))
    lines = [
        f"torch.save({{",
        f'    "epoch": {epoch},',
        f'    "loss": {loss},',
        f'    "model_state_dict": {model}.state_dict() if {model} is not None else None,',
        f'    "optimizer_state_dict": {optimizer}.state_dict() if {optimizer} is not None else None,',
        f"}}, {path})",
    ]
    out_path = ov.get("path")
    if out_path:
        lines.append(f"{out_path} = {path}")
    return ["import torch"], lines


@_reg("pt_load_checkpoint")
def _gen_pt_load_checkpoint(node, iv, ov):
    model     = iv.get("model",     "None")
    optimizer = iv.get("optimizer", "None")
    path      = iv.get("path",      repr(node.inputs["path"].default_value))
    device    = iv.get("device",    repr(node.inputs["device"].default_value))
    out_model = ov.get("model",     "_model")
    out_opt   = ov.get("optimizer", "_optimizer")
    out_epoch = ov.get("epoch",     "_epoch")
    out_loss  = ov.get("loss",      "_loss")
    lines = [
        f"_ckpt = torch.load({path}, map_location={device}, weights_only=False)",
        f"if {model} is not None and 'model_state_dict' in _ckpt:",
        f"    {model}.load_state_dict(_ckpt['model_state_dict'])",
        f"    {model}.to({device})",
        f"if {optimizer} is not None and 'optimizer_state_dict' in _ckpt:",
        f"    {optimizer}.load_state_dict(_ckpt['optimizer_state_dict'])",
        f"{out_model} = {model}",
        f"{out_opt} = {optimizer}",
        f"{out_epoch} = _ckpt.get('epoch', 0)",
        f"{out_loss} = _ckpt.get('loss', 0.0)",
    ]
    return ["import torch"], lines


@_reg("pt_export_onnx")
def _gen_pt_export_onnx(node, iv, ov):
    model  = iv.get("model",       "None")
    path   = iv.get("path",        repr(node.inputs["path"].default_value))
    opset  = iv.get("opset",       "17")
    shape  = iv.get("input_shape", repr(node.inputs["input_shape"].default_value))
    out_path = ov.get("path", "_onnx_path")
    lines = [
        f"_onnx_shape = tuple(int(s.strip()) for s in {shape}.split(','))",
        f"_onnx_dummy = torch.zeros(*_onnx_shape)",
        f"{model}.eval()",
        f"torch.onnx.export({model}, _onnx_dummy, {path},",
        f"    opset_version={opset}, input_names=['input'], output_names=['output'])",
        f"{out_path} = {path}",
    ]
    return ["import torch"], lines


@_reg("pt_save_full_model")
def _gen_pt_save_full_model(node, iv, ov):
    model = iv.get("model", "None")
    path  = _val(node, iv, "path")
    out_m = ov.get("model", "_model_pass")
    lines = [
        f"torch.save({model}, {path})",
        f"{out_m} = {model}",
    ]
    return ["import torch"], lines


@_reg("pt_pretrained_block")
def _gen_pt_pretrained_block(node, iv, ov):
    path             = _val(node, iv, "path")
    device           = _val(node, iv, "device")
    freeze_all       = _val(node, iv, "freeze_all")
    trainable_layers = _val(node, iv, "trainable_layers")
    eval_mode        = _val(node, iv, "eval_mode")
    out_m  = ov.get("model", "_pretrained")
    lines = [
        f"{out_m} = torch.load({path}, map_location={device}, weights_only=False)",
        f"{out_m}.to({device})",
        f"if {freeze_all}:",
        f"    for _p in {out_m}.parameters(): _p.requires_grad = False",
        f"if {trainable_layers} > 0:",
        f"    for _child in list({out_m}.children())[-{trainable_layers}:]:",
        f"        for _p in _child.parameters(): _p.requires_grad = True",
        f"if {eval_mode}: {out_m}.eval()",
    ]
    return ["import torch"], lines


@_reg("pt_model_info_persist")
def _gen_pt_model_info_persist(node, iv, ov):
    model   = iv.get("model", "None")
    out_m   = ov.get("model", "_model_passthrough")
    out_info = ov.get("info", "_model_info")
    lines = [
        f"_total_p = sum(p.numel() for p in {model}.parameters())",
        f"_train_p = sum(p.numel() for p in {model}.parameters() if p.requires_grad)",
        f"{out_info} = f'Params: {{_total_p:,}} total / {{_train_p:,}} trainable'",
        f"print({out_info})",
        f"{out_m} = {model}",
    ]
    return [], lines


# ── PyTorch — architecture ────────────────────────────────────────────────────

@_reg("pt_residual_block")
def _gen_pt_residual_block(node, iv, ov):
    block = iv.get("block", "None")
    proj  = iv.get("projection", "None")
    out   = ov.get("model", "_res_block")
    lines = [
        f"class _ResBlock_{out}(nn.Module):",
        f"    def __init__(self):",
        f"        super().__init__()",
        f"        self.block = {block}",
        f"        self.projection = {proj}",
        f"    def forward(self, x):",
        f"        identity = x if self.projection is None else self.projection(x)",
        f"        return identity + self.block(x)",
        f"{out} = _ResBlock_{out}()",
    ]
    return ["import torch.nn as nn"], lines


@_reg("pt_concat_branches")
def _gen_pt_concat_branches(node, iv, ov):
    branches = [iv.get(f"branch_{i}") for i in range(1, 5) if iv.get(f"branch_{i}")]
    dim  = _val(node, iv, "dim")
    out  = ov.get("model", "_concat_mod")
    branch_attrs = "\n".join(f"        self.branch_{i} = {b}" for i, b in enumerate(branches))
    branch_calls = ", ".join(f"self.branch_{i}(x)" for i in range(len(branches)))
    lines = [
        f"class _ConcatMod_{out}(nn.Module):",
        f"    def __init__(self):",
        f"        super().__init__()",
        branch_attrs,
        f"    def forward(self, x):",
        f"        return torch.cat([{branch_calls}], dim={dim})",
        f"{out} = _ConcatMod_{out}()",
    ]
    return ["import torch", "import torch.nn as nn"], lines


@_reg("pt_add_branches")
def _gen_pt_add_branches(node, iv, ov):
    b1  = iv.get("branch_1", "None")
    b2  = iv.get("branch_2", "None")
    out = ov.get("model", "_add_mod")
    lines = [
        f"class _AddMod_{out}(nn.Module):",
        f"    def __init__(self):",
        f"        super().__init__()",
        f"        self.branch_1 = {b1}",
        f"        self.branch_2 = {b2}",
        f"    def forward(self, x):",
        f"        return self.branch_1(x) + self.branch_2(x)",
        f"{out} = _AddMod_{out}()",
    ]
    return ["import torch.nn as nn"], lines


@_reg("pt_custom_module")
def _gen_pt_custom_module(node, iv, ov):
    code = node.inputs["forward_code"].default_value or "return self.mod_1(x)"
    mods = {f"mod_{i}": iv.get(f"mod_{i}") for i in range(1, 5) if iv.get(f"mod_{i}")}
    out  = ov.get("model", "_custom_mod")
    import textwrap
    indented = textwrap.indent(code.strip(), "        ")
    mod_attrs = "\n".join(f"        self.{k} = {v}" for k, v in mods.items())
    lines = [
        "import torch",
        "import torch.nn as nn",
        "import torch.nn.functional as F",
        f"class _CustomMod_{out}(nn.Module):",
        f"    def __init__(self):",
        f"        super().__init__()",
        mod_attrs,
        f"    def forward(self, x):",
        indented,
        f"{out} = _CustomMod_{out}()",
    ]
    return [], lines


# ── Viz — render matplotlib inline; for export just save to file ──────────────

def _gen_viz_unsupported(node, iv, ov):
    return [], [f"# [{node.label}]: visualization nodes render inline — skipped in export"]

@_reg("ai_ollama_generate")
def _gen_ai_ollama_generate(node, iv, ov):
    prompt = _val(node, iv, "prompt")
    model  = _val(node, iv, "model")
    system = _val(node, iv, "system")
    temp   = _val(node, iv, "temperature")
    maxt   = _val(node, iv, "max_tokens")
    host   = _val(node, iv, "host")
    out    = ov.get("response", "_ollama_response")
    lines  = [
        "import requests as _req",
        f"_ollama_msgs = []",
        f"if {system}: _ollama_msgs.append({{'role': 'system', 'content': {system}}})",
        f"_ollama_msgs.append({{'role': 'user', 'content': {prompt}}})",
        f"_ollama_r = _req.post({host} + '/api/chat', json={{'model': {model}, 'messages': _ollama_msgs, 'stream': False, 'options': {{'temperature': {temp}, 'num_predict': {maxt}}}}}, timeout=120)",
        f"{out} = _ollama_r.json().get('message', {{}}).get('content', '')",
    ]
    return [], lines


@_reg("ai_ollama_embed")
def _gen_ai_ollama_embed(node, iv, ov):
    text  = _val(node, iv, "text")
    model = _val(node, iv, "model")
    host  = _val(node, iv, "host")
    out   = ov.get("embedding", "_ollama_embed")
    lines = [
        "import requests as _req, numpy as _np",
        f"_embed_r = _req.post({host} + '/api/embed', json={{'model': {model}, 'input': {text}}}, timeout=60)",
        f"{out} = _np.array(_embed_r.json().get('embeddings', [[]])[0], dtype=_np.float32)",
    ]
    return [], lines


@_reg("ai_agno_agent")
def _gen_ai_agno_agent(node, iv, ov):
    message  = _val(node, iv, "message")
    agent_id = _val(node, iv, "agent_id")
    host     = _val(node, iv, "host")
    out      = ov.get("response", "_agno_response")
    lines    = [
        "import requests as _req",
        f"_agno_url = {host} + ('/v1/teams/' if 'team' in str({agent_id}) else '/v1/agents/') + str({agent_id}) + '/runs'",
        f"_agno_r = _req.post(_agno_url, json={{'message': {message}}}, timeout=180)",
        f"_agno_d = _agno_r.json()",
        f"{out} = _agno_d.get('content') or _agno_d.get('message', {{}}).get('content') or str(_agno_d)",
    ]
    return [], lines


@_reg("ai_hf_model")
def _gen_ai_hf_model(node, iv, ov):
    name        = _val(node, iv, "model_name")
    num_labels  = _val(node, iv, "num_labels")
    freeze_base = _val(node, iv, "freeze_base")
    device      = _val(node, iv, "device")
    out_m = ov.get("model", "_hf_model")
    out_t = ov.get("tokenizer", "_hf_tokenizer")
    lines = [
        "from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification",
        f"{out_t} = AutoTokenizer.from_pretrained({name})",
        f"if {num_labels} > 0:",
        f"    {out_m} = AutoModelForSequenceClassification.from_pretrained({name}, num_labels={num_labels})",
        f"else:",
        f"    {out_m} = AutoModel.from_pretrained({name})",
        f"{out_m}.to({device})",
        f"if {freeze_base}:",
        f"    for _n, _p in {out_m}.named_parameters():",
        f"        if 'classifier' not in _n and 'cls' not in _n: _p.requires_grad = False",
    ]
    return [], lines


@_reg("ai_hf_tokenize")
def _gen_ai_hf_tokenize(node, iv, ov):
    tok    = iv.get("tokenizer", "_hf_tokenizer")
    text   = _val(node, iv, "text")
    maxlen = _val(node, iv, "max_length")
    device = _val(node, iv, "device")
    out_ids  = ov.get("input_ids",      "_input_ids")
    out_mask = ov.get("attention_mask", "_attn_mask")
    lines = [
        f"_encoding = {tok}({text}, max_length={maxlen}, padding='max_length', truncation=True, return_tensors='pt')",
        f"{out_ids}  = _encoding['input_ids'].to({device})",
        f"{out_mask} = _encoding.get('attention_mask', None)",
        f"if {out_mask} is not None: {out_mask} = {out_mask}.to({device})",
    ]
    return [], lines


@_reg("ai_hf_pipeline")
def _gen_ai_hf_pipeline(node, iv, ov):
    task    = _val(node, iv, "task")
    model   = _val(node, iv, "model_name")
    text    = _val(node, iv, "input_text")
    maxlen  = _val(node, iv, "max_length")
    out     = ov.get("output", "_pipeline_output")
    lines   = [
        "from transformers import pipeline as _hf_pipeline",
        f"_pipe = _hf_pipeline({task}, model={model})",
        f"_pipe_result = _pipe({text}, max_length={maxlen}, truncation=True)",
        f"{out} = _pipe_result[0].get('generated_text') or str(_pipe_result[0]) if _pipe_result else ''",
    ]
    return [], lines


for _vt in ("viz_line","viz_scatter","viz_bar","viz_hist","viz_heatmap",
            "viz_box","viz_conf_matrix","viz_pca_2d","viz_loss_curve","viz_image_grid",
            "pt_viz_tensor_heatmap","pt_viz_training_curve","pt_viz_tensor_hist",
            "pt_viz_tensor_scatter","pt_viz_show_image","pt_viz_weight_hist",
            "io_serial_out","io_serial_in","io_serial_list",
            "io_http_post","io_mqtt_publish","io_websocket_send","io_ros_publish",
            "io_csv_writer","io_json_writer","io_text_log"):
    _GENERATORS[_vt] = _gen_viz_unsupported


# ── Exporter ──────────────────────────────────────────────────────────────────

class GraphExporter:
    """Export a Graph to a runnable Python script."""

    def export(self, graph: Graph) -> str:
        order = graph.topological_order()
        if not order:
            return "# Empty graph — nothing to export.\n"

        # Build connection lookup: (to_node_id, to_port) -> (from_node_id, from_port)
        conn_map: dict[tuple[str, str], tuple[str, str]] = {}
        for c in graph.connections:
            conn_map[(c.to_node_id, c.to_port)] = (c.from_node_id, c.from_port)

        # Variable name counters per prefix
        counters: dict[str, int] = {}
        # Map (node_id, port_name) -> variable name for each output port
        var_map: dict[tuple[str, str], str] = {}

        def _make_var(type_name: str, port_name: str) -> str:
            prefix = _short(type_name)
            # Use port name when there are multiple outputs to keep names readable
            node = graph.nodes.get(type_name)  # not available here, use port_name suffix
            key = prefix
            counters[key] = counters.get(key, 0)
            name = f"{prefix}_{counters[key]}"
            counters[key] += 1
            return name

        # Assign output variable names — one per output port
        port_counters: dict[str, int] = {}

        def assign_out_vars(node: BaseNode) -> dict[str, str]:
            out_vars: dict[str, str] = {}
            prefix = _short(node.type_name)
            for port_name in node.outputs:
                if port_name == "__terminal__":
                    continue
                key = f"{prefix}_{port_name}"
                port_counters[key] = port_counters.get(key, 0)
                vname = f"{prefix}_{port_counters[key]}" if len(node.outputs) <= 1 else f"{prefix}_{port_name}_{port_counters[key]}"
                port_counters[key] += 1
                out_vars[port_name] = vname
                var_map[(node.id, port_name)] = vname
            return out_vars

        all_imports: list[str] = []
        all_blocks: list[str] = []

        for node_id in order:
            node = graph.nodes[node_id]

            # Build in_vars: port_name -> variable expression (or None)
            in_vars: dict[str, str | None] = {}
            for port_name in node.inputs:
                key = (node_id, port_name)
                if key in conn_map:
                    from_node_id, from_port = conn_map[key]
                    var = var_map.get((from_node_id, from_port))
                    in_vars[port_name] = var  # may still be None if upstream had no output
                else:
                    in_vars[port_name] = None  # use default

            out_vars = assign_out_vars(node)

            gen = _GENERATORS.get(node.type_name)
            if gen is None:
                all_blocks.append(f"# [{node.label}]: export not yet supported")
                continue

            try:
                imports, lines = gen(node, in_vars, out_vars)
                all_imports.extend(imports)
                if lines:
                    all_blocks.append(f"# {node.label}")
                    all_blocks.extend(lines)
                    all_blocks.append("")
            except Exception as exc:
                all_blocks.append(f"# [{node.label}]: export error — {exc}")

        # Deduplicate imports, keeping order
        seen: set[str] = set()
        deduped_imports: list[str] = []
        for imp in all_imports:
            if imp not in seen:
                seen.add(imp)
                deduped_imports.append(imp)

        # Group imports: stdlib → third-party
        stdlib = [i for i in deduped_imports if i.startswith(("import math", "import random", "import json"))]
        third_party = [i for i in deduped_imports if i not in stdlib]

        header = ["# Generated by Node Tool v3", ""]
        import_block = stdlib + ([""] if stdlib and third_party else []) + third_party

        sections = header + import_block + ([""] if import_block else []) + all_blocks
        return "\n".join(sections).rstrip() + "\n"
