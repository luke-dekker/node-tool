"""NumPy nodes — consolidated 12 → 3.

Active:
- NpCreateNode  — creator (no array input). kind: arange|linspace|ones|zeros|eye|rand|randn|from_list
- NpOpNode      — op dropdown over transforms / reductions / element-wise / inspections
- NpLinalgNode  — multi-output linear algebra (inv|svd|eig|dot|matmul)

All old per-op class names alias to one of the three. Caller sets `op`/`kind`
on the instance to recover specific behavior.
"""
from nodes.numpy.np_create import NpCreateNode
from nodes.numpy.np_op     import NpOpNode
from nodes.numpy.np_linalg import NpLinalgNode

# Back-compat — NpCreateNode kinds
NpArangeNode    = NpCreateNode
NpLinspaceNode  = NpCreateNode
NpZerosNode     = NpCreateNode
NpOnesNode      = NpCreateNode
NpRandNode      = NpCreateNode
NpRandnNode     = NpCreateNode
NpFromListNode  = NpCreateNode
NpEyeNode       = NpCreateNode

# Back-compat — NpLinalgNode kinds
NpDotNode    = NpLinalgNode
NpMatMulNode = NpLinalgNode
NpInvNode    = NpLinalgNode
NpEigNode    = NpLinalgNode
NpSVDNode    = NpLinalgNode

# Back-compat — NpOpNode ops (transforms / reductions / element-wise / inspect)
NpClipNode      = NpOpNode  # op="clip"
NpConcatNode    = NpOpNode  # op="concat"
NpStackNode     = NpOpNode  # op="stack"
NpSliceNode     = NpOpNode  # op="slice"
NpWhereNode     = NpOpNode  # op="where"
NpReshapeNode   = NpOpNode  # op="reshape"
NpShapeNode     = NpOpNode  # op="shape"
NpArrayInfoNode = NpOpNode  # op="info"
NpArrayFuncNode = NpOpNode  # op in {abs,sqrt,log,exp,transpose,flatten,normalize,sign,cumsum,diff}
NpReduceNode    = NpOpNode  # op in {sum,mean,std,var,min,max,median,prod,any,all}
