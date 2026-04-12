"""HF Tokenizer Transform node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class HFTokenizerTransformNode(BaseNode):
    type_name   = "pt_hf_tokenizer_transform"
    label       = "HF Tokenizer"
    category    = "Data"
    subcategory = "Transforms"
    description = "Tokenize text using a HuggingFace tokenizer (e.g. 'bert-base-uncased'). Outputs a callable transform."

    def _setup_ports(self):
        self.add_input("model_name",  PortType.STRING, default="bert-base-uncased")
        self.add_input("max_length",  PortType.INT,    default=128)
        self.add_input("padding",     PortType.BOOL,   default=True)
        self.add_input("truncation",  PortType.BOOL,   default=True)
        self.add_output("transform", PortType.TRANSFORM)

    def execute(self, inputs):
        try:
            from transformers import AutoTokenizer
            name     = str(inputs.get("model_name") or "bert-base-uncased")
            max_len  = int(inputs.get("max_length") or 128)
            padding  = bool(inputs.get("padding", True))
            truncate = bool(inputs.get("truncation", True))
            tok = AutoTokenizer.from_pretrained(name)

            class TokenizerTransform:
                def __call__(self, text):
                    import torch
                    enc = tok(text, max_length=max_len, padding="max_length" if padding else False,
                              truncation=truncate, return_tensors="pt")
                    return {k: v.squeeze(0) for k, v in enc.items()}

            return {"transform": TokenizerTransform()}
        except Exception:
            return {"transform": None}

    def export(self, iv, ov):
        name = self._val(iv, 'model_name'); ml = self._val(iv, 'max_length')
        tfv = ov['transform']
        return ["from transformers import AutoTokenizer"], [
            f"_tok_{tfv} = AutoTokenizer.from_pretrained({name})",
            f"{tfv} = lambda text: _tok_{tfv}(text, max_length={ml}, padding='max_length', truncation=True, return_tensors='pt')",
        ]
