"""HF Pipeline node."""
from __future__ import annotations
from typing import Any
from core.node import BaseNode, PortType


class HFPipelineNode(BaseNode):
    type_name   = "ai_hf_pipeline"
    label       = "HF Pipeline"
    category    = "AI"
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

    def export(self, iv, ov):
        task   = self._val(iv, "task")
        model  = self._val(iv, "model_name")
        text   = self._val(iv, "input_text")
        maxlen = self._val(iv, "max_length")
        out    = ov.get("output", "_pipeline_output")
        lines  = [
            "from transformers import pipeline as _hf_pipeline",
            f"_pipe = _hf_pipeline({task}, model={model})",
            f"_pipe_result = _pipe({text}, max_length={maxlen}, truncation=True)",
            f"{out} = _pipe_result[0].get('generated_text') or str(_pipe_result[0]) if _pipe_result else ''",
        ]
        return [], lines
