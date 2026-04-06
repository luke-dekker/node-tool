"""HuggingFace Model node."""
from __future__ import annotations
from typing import Any
from core.node import BaseNode, PortType


class HuggingFaceModelNode(BaseNode):
    type_name   = "ai_hf_model"
    label       = "HF Model"
    category    = "Models"
    subcategory = "Pretrained"
    description = (
        "Download any HuggingFace model as a real nn.Module. "
        "Wire into Pretrained Block pattern: freeze it, add a head, fine-tune. "
        "Requires: pip install transformers"
    )

    def _setup_ports(self) -> None:
        self.add_input("model_name",  PortType.STRING, default="distilbert-base-uncased",
                       description="HF Hub model id, e.g. bert-base-uncased, gpt2, distilbert-base-uncased")
        self.add_input("num_labels",  PortType.INT,    default=0,
                       description="For classification heads: number of output labels (0 = base model, no head)")
        self.add_input("freeze_base", PortType.BOOL,   default=False,
                       description="Freeze all base model weights (train only the head)")
        self.add_input("device",      PortType.STRING, default="cpu")
        self.add_input("cache_dir",   PortType.STRING, default="",
                       description="Local cache directory (blank = HF default ~/.cache/huggingface)")
        self.add_output("model",     PortType.MODULE)
        self.add_output("tokenizer", PortType.ANY,
                        description="Tokenizer — wire into HF Tokenize node")
        self.add_output("info",      PortType.STRING)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        model_name  = inputs.get("model_name")  or "distilbert-base-uncased"
        num_labels  = int(inputs.get("num_labels")  or 0)
        freeze_base = bool(inputs.get("freeze_base", False))
        device      = inputs.get("device") or "cpu"
        cache_dir   = inputs.get("cache_dir") or None

        try:
            from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
            import torch

            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

            if num_labels > 0:
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name, num_labels=num_labels, cache_dir=cache_dir
                )
            else:
                model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)

            model.to(device)

            if freeze_base:
                # Freeze everything except classification head (if present)
                for name, param in model.named_parameters():
                    if "classifier" not in name and "cls" not in name:
                        param.requires_grad = False

            total     = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            info = (
                f"{model.__class__.__name__} loaded from HuggingFace\n"
                f"Model: {model_name}\n"
                f"Device: {device}\n"
                f"Total:     {total:,} params\n"
                f"Trainable: {trainable:,} params\n"
                f"Frozen:    {total - trainable:,} params"
            )
            return {"model": model, "tokenizer": tokenizer, "info": info}

        except ImportError:
            return {
                "model": None, "tokenizer": None,
                "info": "transformers not installed — run: pip install transformers",
            }
        except Exception as exc:
            return {"model": None, "tokenizer": None, "info": f"Load failed: {exc}"}

    def export(self, iv, ov):
        name        = self._val(iv, "model_name")
        num_labels  = self._val(iv, "num_labels")
        freeze_base = self._val(iv, "freeze_base")
        device      = self._val(iv, "device")
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
