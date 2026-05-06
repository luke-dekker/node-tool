"""Pandas nodes — consolidated to 3 mega-nodes.

- PdSourceNode    — kind: csv | json | numpy | sample → DataFrame
- PdTransformNode — every df→? op: transforms / info / extract (16 ops)
- PdMergeNode     — multi-input merge (kept; distinct semantics)

Every old class name aliases to one of the three. Caller sets `kind`/`op`
on the instance to recover specific behavior.
"""
from nodes.pandas.pd_source    import PdSourceNode
from nodes.pandas.pd_transform import PdTransformNode
from nodes.pandas.pd_merge     import PdMergeNode

# Back-compat — PdSourceNode kinds
PdFromCsvNode      = PdSourceNode
PdFromDictNode     = PdSourceNode
PdFromNumpyNode    = PdSourceNode
PdMakeSampleNode   = PdSourceNode    # kind="sample"

# Back-compat — PdTransformNode ops (transforms)
PdDropNaNode       = PdTransformNode
PdFillNaNode       = PdTransformNode
PdDropColsNode     = PdTransformNode
PdSelectColsNode   = PdTransformNode
PdRenameColNode    = PdTransformNode
PdResetIndexNode   = PdTransformNode
PdNormalizeNode    = PdTransformNode
PdSortNode         = PdTransformNode
PdFilterRowsNode   = PdTransformNode
PdCorrelationNode  = PdTransformNode
PdGroupByNode      = PdTransformNode

# Back-compat — PdTransformNode info ops (was PdInfoNode)
PdInfoNode         = PdTransformNode
PdDescribeNode     = PdTransformNode
PdHeadNode         = PdTransformNode
PdShapeNode        = PdTransformNode

# Back-compat — PdTransformNode extract ops
PdGetColumnNode    = PdTransformNode
PdToNumpyNode      = PdTransformNode
PdXYSplitNode      = PdTransformNode

__all__ = [
    "PdSourceNode", "PdTransformNode", "PdMergeNode",
    "PdFromCsvNode", "PdFromDictNode", "PdFromNumpyNode", "PdMakeSampleNode",
    "PdDropNaNode", "PdFillNaNode", "PdDropColsNode", "PdSelectColsNode",
    "PdRenameColNode", "PdResetIndexNode", "PdNormalizeNode", "PdSortNode",
    "PdFilterRowsNode", "PdCorrelationNode", "PdGroupByNode",
    "PdInfoNode", "PdDescribeNode", "PdHeadNode", "PdShapeNode",
    "PdGetColumnNode", "PdToNumpyNode", "PdXYSplitNode",
]
