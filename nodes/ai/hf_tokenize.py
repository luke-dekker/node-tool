"""HF Tokenize node."""
from __future__ import annotations
from typing import Any
from core.node import BaseNode, PortType


class HFTokenizeNode(BaseNode):
    type_name   = "ai_hf_tokenize"
    label       = "HF Tokenize"
    category    = "Models"
    subcategory = "Pretrained"
    description = (
        "Tokenize text using a HuggingFace tokenizer. "
        "Outputs input_ids tensor ready for the model."
    )

    def _setup_ports(self) -> None:
        self.add_input("tokenizer",   PortType.ANY,    default=None,
                       description="Tokenizer from HF Model node")
        self.add_input("text",        PortType.STRING, default="",
                       description="Text to tokenize")
        self.add_input("max_length",  PortType.INT,    default=512)
        self.add_input("device",      PortType.STRING, default="cpu")
        self.add_output("input_ids",      PortType.TENSOR,
                        description="Token ID tensor — feed directly into model")
        self.add_output("attention_mask", PortType.TENSOR)
        self.add_output("encoding",       PortType.ANY,
                        description="Full encoding dict for advanced use")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        tokenizer  = inputs.get("tokenizer")
        text       = inputs.get("text") or ""
        max_length = int(inputs.get("max_length") or 512)
        device     = inputs.get("device") or "cpu"

        null = {"input_ids": None, "attention_mask": None, "encoding": None}

        if tokenizer is None:
            return null
        if not text:
            return null

        try:
            import torch
            encoding = tokenizer(
                text,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids      = encoding["input_ids"].to(device)
            attention_mask = encoding.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            return {
                "input_ids":      input_ids,
                "attention_mask": attention_mask,
                "encoding":       {k: v.to(device) for k, v in encoding.items()},
            }
        except Exception as exc:
            return null

    def export(self, iv, ov):
        tok    = iv.get("tokenizer", "_hf_tokenizer")
        text   = self._val(iv, "text")
        maxlen = self._val(iv, "max_length")
        device = self._val(iv, "device")
        out_ids  = ov.get("input_ids", "_input_ids")
        out_mask = ov.get("attention_mask", "_attn_mask")
        lines = [
            f"_encoding = {tok}({text}, max_length={maxlen}, padding='max_length', truncation=True, return_tensors='pt')",
            f"{out_ids}  = _encoding['input_ids'].to({device})",
            f"{out_mask} = _encoding.get('attention_mask', None)",
            f"if {out_mask} is not None: {out_mask} = {out_mask}.to({device})",
        ]
        return [], lines
