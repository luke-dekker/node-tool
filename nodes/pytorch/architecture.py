"""Re-export shim — BranchOpNode absorbs Add/Concat branches via op dropdown."""
from nodes.pytorch.residual_block import ResidualBlockNode
from nodes.pytorch.branch_op      import BranchOpNode
from nodes.pytorch.custom_module  import CustomModuleNode

# Back-compat aliases
ConcatBranchesNode = BranchOpNode  # op="concat"
AddBranchesNode    = BranchOpNode  # op="add"

__all__ = [
    "ResidualBlockNode", "BranchOpNode", "CustomModuleNode",
    "ConcatBranchesNode", "AddBranchesNode",
]
