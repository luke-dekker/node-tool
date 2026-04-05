"""Tests for AI inference nodes (Ollama, Agno, HuggingFace).

Ollama/Agno tests mock the HTTP layer — no live server needed.
HuggingFace tests are skipped if transformers is not installed.
"""
from __future__ import annotations
import pytest
from unittest.mock import patch, MagicMock


# ── OllamaGenerateNode ────────────────────────────────────────────────────────

class TestOllamaGenerateNode:
    def _node(self):
        from nodes.ai.ollama_nodes import OllamaGenerateNode
        return OllamaGenerateNode()

    def test_no_prompt_returns_empty(self):
        n = self._node()
        r = n.execute({"prompt": "", "model": "m", "system": "",
                        "temperature": 0.7, "max_tokens": 10,
                        "host": "http://localhost:11434"})
        assert r["response"] == ""
        assert "No prompt" in r["__terminal__"]

    def test_successful_call(self):
        n = self._node()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "Hello world"}}
        mock_resp.raise_for_status = MagicMock()
        with patch("requests.post", return_value=mock_resp):
            r = n.execute({"prompt": "Hi", "model": "qwen3:32b",
                            "system": "", "temperature": 0.7,
                            "max_tokens": 100, "host": "http://localhost:11434"})
        assert r["response"] == "Hello world"
        assert "qwen3:32b" in r["__terminal__"]

    def test_with_system_prompt(self):
        n = self._node()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "ok"}}
        mock_resp.raise_for_status = MagicMock()
        captured = {}
        def capture_post(url, json=None, **kw):
            captured["json"] = json
            return mock_resp
        with patch("requests.post", side_effect=capture_post):
            n.execute({"prompt": "Hi", "model": "m", "system": "Be terse.",
                        "temperature": 0.7, "max_tokens": 10,
                        "host": "http://localhost:11434"})
        msgs = captured["json"]["messages"]
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "Be terse."

    def test_network_error_returns_error_string(self):
        n = self._node()
        with patch("requests.post", side_effect=ConnectionError("refused")):
            r = n.execute({"prompt": "Hi", "model": "m", "system": "",
                            "temperature": 0.7, "max_tokens": 10,
                            "host": "http://localhost:11434"})
        assert r["response"] == ""
        assert "Error" in r["__terminal__"]


# ── OllamaEmbedNode ───────────────────────────────────────────────────────────

class TestOllamaEmbedNode:
    def _node(self):
        from nodes.ai.ollama_nodes import OllamaEmbedNode
        return OllamaEmbedNode()

    def test_no_text_returns_none(self):
        n = self._node()
        r = n.execute({"text": "", "model": "nomic-embed-text",
                        "host": "http://localhost:11434"})
        assert r["embedding"] is None

    def test_returns_numpy_array(self):
        import numpy as np
        n = self._node()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
        mock_resp.raise_for_status = MagicMock()
        with patch("requests.post", return_value=mock_resp):
            r = n.execute({"text": "hello", "model": "nomic-embed-text",
                            "host": "http://localhost:11434"})
        assert isinstance(r["embedding"], np.ndarray)
        assert r["embedding"].shape == (3,)
        assert r["embedding"].dtype.name == "float32"

    def test_error_returns_none(self):
        n = self._node()
        with patch("requests.post", side_effect=Exception("fail")):
            r = n.execute({"text": "hi", "model": "m",
                            "host": "http://localhost:11434"})
        assert r["embedding"] is None
        assert "Error" in r["__terminal__"]


# ── AgnoAgentNode ─────────────────────────────────────────────────────────────

class TestAgnoAgentNode:
    def _node(self):
        from nodes.ai.ollama_nodes import AgnoAgentNode
        return AgnoAgentNode()

    def test_no_message_returns_empty(self):
        n = self._node()
        r = n.execute({"message": "", "agent_id": "coding-team",
                        "host": "http://localhost:8000", "stream": False})
        assert r["response"] == ""

    def test_team_uses_teams_endpoint(self):
        n = self._node()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"content": "done"}
        mock_resp.raise_for_status = MagicMock()
        captured = {}
        def capture(url, **kw):
            captured["url"] = url
            return mock_resp
        with patch("requests.post", side_effect=capture):
            n.execute({"message": "hi", "agent_id": "coding-team",
                        "host": "http://localhost:8000", "stream": False})
        assert "/teams/" in captured["url"]

    def test_agent_uses_agents_endpoint(self):
        n = self._node()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"content": "done"}
        mock_resp.raise_for_status = MagicMock()
        captured = {}
        def capture(url, **kw):
            captured["url"] = url
            return mock_resp
        with patch("requests.post", side_effect=capture):
            n.execute({"message": "hi", "agent_id": "coder",
                        "host": "http://localhost:8000", "stream": False})
        assert "/agents/" in captured["url"]

    def test_error_handled_gracefully(self):
        n = self._node()
        with patch("requests.post", side_effect=Exception("timeout")):
            r = n.execute({"message": "hi", "agent_id": "coder",
                            "host": "http://localhost:8000", "stream": False})
        assert r["response"] == ""
        assert "Error" in r["__terminal__"]


# ── HuggingFace nodes — skip if transformers not installed ────────────────────

try:
    import transformers as _transformers_check
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False

hf_only = pytest.mark.skipif(not _HF_AVAILABLE, reason="transformers not installed")


@hf_only
class TestHFTokenizeNode:
    def _node(self):
        from nodes.ai.hf_nodes import HFTokenizeNode
        return HFTokenizeNode()

    def test_no_tokenizer_returns_none(self):
        n = self._node()
        r = n.execute({"tokenizer": None, "text": "hello",
                        "max_length": 32, "device": "cpu"})
        assert r["input_ids"] is None

    def test_no_text_returns_none(self):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        n = self._node()
        r = n.execute({"tokenizer": tok, "text": "",
                        "max_length": 32, "device": "cpu"})
        assert r["input_ids"] is None

    def test_tokenizes_text(self):
        import torch
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        n = self._node()
        r = n.execute({"tokenizer": tok, "text": "Hello world",
                        "max_length": 32, "device": "cpu"})
        assert isinstance(r["input_ids"], torch.Tensor)
        assert r["input_ids"].shape[1] == 32
        assert r["attention_mask"] is not None


@hf_only
class TestHFModelNode:
    def _node(self):
        from nodes.ai.hf_nodes import HuggingFaceModelNode
        return HuggingFaceModelNode()

    def test_loads_base_model(self):
        import torch.nn as nn
        n = self._node()
        r = n.execute({"model_name": "distilbert-base-uncased",
                        "num_labels": 0, "freeze_base": False,
                        "device": "cpu", "cache_dir": ""})
        assert isinstance(r["model"], nn.Module)
        assert r["tokenizer"] is not None
        assert "distilbert" in r["info"].lower() or "DistilBert" in r["info"]

    def test_freeze_base(self):
        n = self._node()
        r = n.execute({"model_name": "distilbert-base-uncased",
                        "num_labels": 0, "freeze_base": True,
                        "device": "cpu", "cache_dir": ""})
        model = r["model"]
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        assert trainable < total

    def test_info_shows_param_counts(self):
        n = self._node()
        r = n.execute({"model_name": "distilbert-base-uncased",
                        "num_labels": 0, "freeze_base": False,
                        "device": "cpu", "cache_dir": ""})
        assert "Total:" in r["info"]
        assert "Trainable:" in r["info"]
