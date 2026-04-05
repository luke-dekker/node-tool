"""HuggingFace model and tokenizer nodes."""

from __future__ import annotations
from typing import Any
from core.node import BaseNode
from core.node import PortType

CATEGORY = "AI"
_MODELS_CAT = "Models"


# ── HuggingFace Model ──────────────────────────────────────────────────────────

class HuggingFaceModelNode(BaseNode):
    type_name   = "ai_hf_model"
    label       = "HF Model"
    category    = _MODELS_CAT
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


# ── HF Tokenize ────────────────────────────────────────────────────────────────

class HFTokenizeNode(BaseNode):
    type_name   = "ai_hf_tokenize"
    label       = "HF Tokenize"
    category    = _MODELS_CAT
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


# ── HF Pipeline ────────────────────────────────────────────────────────────────

class HFPipelineNode(BaseNode):
    type_name   = "ai_hf_pipeline"
    label       = "HF Pipeline"
    category    = CATEGORY
    subcategory = "HuggingFace"
    description = (
        "High-level HuggingFace pipeline for inference. "
        "Good for quick inference without building a full graph. "
        "Tasks: text-generation, sentiment-analysis, zero-shot-classification, "
        "text2text-generation, summarization, translation, fill-mask, ner"
    )

    def _setup_ports(self) -> None:
        self.add_input("task",       PortType.STRING, default="text-generation",
                       description="Pipeline task name")
        self.add_input("model_name", PortType.STRING, default="gpt2",
                       description="HF Hub model id (blank = default for task)")
        self.add_input("input_text", PortType.STRING, default="",
                       description="Text to process")
        self.add_input("max_length", PortType.INT,    default=100)
        self.add_input("device",     PortType.STRING, default="cpu",
                       description="cpu or cuda:0")
        self.add_output("output",       PortType.STRING,
                        description="Pipeline result as string")
        self.add_output("raw",          PortType.ANY,
                        description="Raw pipeline output (list/dict)")
        self.add_output("__terminal__", PortType.STRING)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        task       = inputs.get("task")       or "text-generation"
        model_name = inputs.get("model_name") or None
        text       = inputs.get("input_text") or ""
        max_length = int(inputs.get("max_length") or 100)
        device_str = inputs.get("device") or "cpu"

        null = {"output": "", "raw": None, "__terminal__": ""}

        if not text:
            return {**null, "__terminal__": "[HF Pipeline] No input text."}

        try:
            from transformers import pipeline

            device = 0 if device_str.startswith("cuda") else -1
            kwargs: dict[str, Any] = {"device": device}
            if model_name:
                kwargs["model"] = model_name

            pipe = pipeline(task, **kwargs)
            result = pipe(text, max_length=max_length, truncation=True)

            # Flatten result to a readable string
            if isinstance(result, list) and result:
                first = result[0]
                if isinstance(first, dict):
                    output = (
                        first.get("generated_text")
                        or first.get("summary_text")
                        or first.get("translation_text")
                        or str(first)
                    )
                else:
                    output = str(first)
            else:
                output = str(result)

            log = f"[HF Pipeline] {task} ({model_name or 'default'}) → {len(output)} chars"
            return {"output": output, "raw": result, "__terminal__": log}

        except ImportError:
            return {**null, "__terminal__": "[HF Pipeline] transformers not installed — pip install transformers"}
        except Exception as exc:
            return {**null, "__terminal__": f"[HF Pipeline] Error: {exc}"}
