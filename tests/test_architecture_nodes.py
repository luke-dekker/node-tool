"""Tests for non-sequential architecture nodes."""
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from nodes.pytorch.architecture import (
    ResidualBlockNode,
    ConcatBranchesNode,
    AddBranchesNode,
    CustomModuleNode,
)


def _linear(in_f, out_f):
    return nn.Linear(in_f, out_f)


# ── ResidualBlockNode ─────────────────────────────────────────────────────────

class TestResidualBlockNode:
    def test_no_block_returns_none(self):
        n = ResidualBlockNode()
        assert n.execute({"block": None, "projection": None})["model"] is None

    def test_basic_residual_forward(self):
        block = nn.Identity()
        n = ResidualBlockNode()
        model = n.execute({"block": block, "projection": None})["model"]
        assert model is not None
        x = torch.randn(2, 4)
        out = model(x)
        assert out.shape == x.shape
        assert torch.allclose(out, x + x)  # identity(x) + x = 2x

    def test_with_projection(self):
        block = nn.Linear(4, 8)
        proj  = nn.Linear(4, 8)
        n = ResidualBlockNode()
        model = n.execute({"block": block, "projection": proj})["model"]
        x = torch.randn(2, 4)
        out = model(x)
        assert out.shape == (2, 8)

    def test_no_projection_no_dim_mismatch(self):
        block = nn.Linear(4, 4)
        n = ResidualBlockNode()
        model = n.execute({"block": block, "projection": None})["model"]
        x = torch.randn(2, 4)
        out = model(x)
        assert out.shape == (2, 4)

    def test_passthrough_type(self):
        n = ResidualBlockNode()
        model = n.execute({"block": nn.Linear(4, 4), "projection": None})["model"]
        assert isinstance(model, nn.Module)


# ── ConcatBranchesNode ────────────────────────────────────────────────────────

class TestConcatBranchesNode:
    def test_no_branches_returns_none(self):
        n = ConcatBranchesNode()
        assert n.execute({"branch_1": None, "branch_2": None,
                          "branch_3": None, "branch_4": None, "dim": 1})["model"] is None

    def test_two_branches_concat(self):
        n = ConcatBranchesNode()
        b1 = nn.Linear(4, 8)
        b2 = nn.Linear(4, 8)
        model = n.execute({"branch_1": b1, "branch_2": b2,
                           "branch_3": None, "branch_4": None, "dim": 1})["model"]
        x = torch.randn(2, 4)
        out = model(x)
        assert out.shape == (2, 16)

    def test_four_branches(self):
        n = ConcatBranchesNode()
        branches = {f"branch_{i}": nn.Linear(4, 4) for i in range(1, 5)}
        branches["dim"] = 1
        model = n.execute(branches)["model"]
        x = torch.randn(2, 4)
        out = model(x)
        assert out.shape == (2, 16)

    def test_single_branch(self):
        n = ConcatBranchesNode()
        model = n.execute({"branch_1": nn.Linear(4, 6), "branch_2": None,
                           "branch_3": None, "branch_4": None, "dim": 1})["model"]
        x = torch.randn(2, 4)
        assert model(x).shape == (2, 6)


# ── AddBranchesNode ───────────────────────────────────────────────────────────

class TestAddBranchesNode:
    def test_missing_branch_returns_none(self):
        n = AddBranchesNode()
        assert n.execute({"branch_1": nn.Linear(4, 4), "branch_2": None})["model"] is None
        assert n.execute({"branch_1": None, "branch_2": nn.Linear(4, 4)})["model"] is None

    def test_add_two_branches(self):
        n = AddBranchesNode()
        # Use Identity so outputs are deterministic
        model = n.execute({"branch_1": nn.Identity(), "branch_2": nn.Identity()})["model"]
        x = torch.randn(2, 4)
        out = model(x)
        assert out.shape == x.shape
        assert torch.allclose(out, x + x)

    def test_returns_module(self):
        n = AddBranchesNode()
        model = n.execute({"branch_1": nn.Linear(4, 4), "branch_2": nn.Linear(4, 4)})["model"]
        assert isinstance(model, nn.Module)


# ── CustomModuleNode ──────────────────────────────────────────────────────────

class TestCustomModuleNode:
    def test_default_code(self):
        n = CustomModuleNode()
        mod = nn.Linear(4, 4)
        result = n.execute({"forward_code": "return self.mod_1(x)",
                            "mod_1": mod, "mod_2": None, "mod_3": None, "mod_4": None})
        assert result["model"] is not None
        assert result["error"] == ""

    def test_forward_runs(self):
        n = CustomModuleNode()
        mod = nn.Linear(4, 8)
        model = n.execute({"forward_code": "return self.mod_1(x)",
                           "mod_1": mod, "mod_2": None, "mod_3": None, "mod_4": None})["model"]
        x = torch.randn(2, 4)
        assert model(x).shape == (2, 8)

    def test_skip_connection_in_code(self):
        n = CustomModuleNode()
        mod = nn.Identity()
        model = n.execute({
            "forward_code": "return x + self.mod_1(x)",
            "mod_1": mod, "mod_2": None, "mod_3": None, "mod_4": None,
        })["model"]
        x = torch.randn(2, 4)
        out = model(x)
        assert torch.allclose(out, x + x)

    def test_multi_module_code(self):
        n = CustomModuleNode()
        m1 = nn.Linear(4, 8)
        m2 = nn.ReLU()
        model = n.execute({
            "forward_code": "return self.mod_2(self.mod_1(x))",
            "mod_1": m1, "mod_2": m2, "mod_3": None, "mod_4": None,
        })["model"]
        x = torch.randn(2, 4)
        out = model(x)
        assert out.shape == (2, 8)
        assert (out >= 0).all()

    def test_syntax_error_returns_error_string(self):
        n = CustomModuleNode()
        result = n.execute({
            "forward_code": "this is not valid python !!!",
            "mod_1": None, "mod_2": None, "mod_3": None, "mod_4": None,
        })
        assert result["model"] is None
        assert len(result["error"]) > 0

    def test_no_mods_still_works(self):
        n = CustomModuleNode()
        model = n.execute({
            "forward_code": "return x * 2",
            "mod_1": None, "mod_2": None, "mod_3": None, "mod_4": None,
        })["model"]
        x = torch.randn(2, 4)
        assert torch.allclose(model(x), x * 2)
