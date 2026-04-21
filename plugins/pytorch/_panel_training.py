"""Training panel spec — the single source of truth for what the training
panel contains. DPG, React, and Godot all render this exact spec.

If you want to change the training panel layout, edit it here. Do NOT edit
gui/mixins/layout.py or any per-GUI code — the GUI is only a renderer.
"""
from __future__ import annotations

from core.panel import (
    PanelSpec, StatusSection, DynamicFormSection, FormSection,
    ButtonsSection, CustomSection, Field, Action,
)
from plugins.pytorch._factories import OPTIMIZER_CHOICES, LOSS_CHOICES


def _detect_devices() -> tuple[list[str], str]:
    """Return (choices, default). Enumerates real CUDA devices by name so the
    dropdown shows 'cuda:0 (RTX 4070)' instead of a hardcoded guess."""
    choices = ["cpu"]
    default = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            for i in range(n):
                try:
                    name = torch.cuda.get_device_name(i)
                except Exception:
                    name = f"cuda:{i}"
                choices.append(f"cuda:{i} ({name})")
            if choices[1:]:
                default = choices[1]
    except Exception:
        pass
    return choices, default


def build_training_panel_spec() -> PanelSpec:
    device_choices, device_default = _detect_devices()
    return PanelSpec(
        label="Training",
        sections=[
            StatusSection(
                id="status",
                label="Status",
                source_rpc="get_training_state",
                poll_ms=500,
                fields=[
                    Field("status",    "str", label="State"),
                    Field("epoch_str", "str", label="Epoch"),
                    Field("best_loss", "str", label="Best loss"),
                    Field("last_loss", "str", label="Last loss"),
                    Field("error",     "str", label="Error"),
                ],
            ),
            CustomSection(
                id="loss_plot",
                label="Loss",
                custom_kind="loss_plot",
                params={
                    "source_rpc": "get_training_losses",
                    "poll_ms":    500,
                    "series":     ["train", "val"],
                },
            ),
            DynamicFormSection(
                id="datasets",
                label="Datasets",
                source_rpc="get_marker_groups",
                item_label_template="[{key}] {modalities}",
                empty_hint="Load a template or add Data In (A) markers to configure datasets.",
                fields=[
                    Field("path",       "str",    label="path",
                          hint="mnist, cifar10, lerobot/...",
                          default=""),
                    Field("batch_size", "int",    label="batch",  default=128, min=1),
                    Field("split",      "choice", label="split",  default="train",
                          choices=["train", "test", "val"],
                          hint="source split loaded from the dataset itself"),
                    Field("val_fraction", "float", label="val%",  default=0.1,
                          min=0.0, max=0.5, step=0.05,
                          hint="0.0 = no validation; otherwise hold out this fraction "
                               "of the loaded dataset as val (stable seed)"),
                    Field("seq_len",    "int",    label="seq",    default=0, min=0),
                    Field("chunk_size", "int",    label="chunk",  default=1, min=1),
                ],
            ),
            FormSection(
                id="hyperparams",
                label="Hyperparameters",
                fields=[
                    Field("epochs",    "int",    label="epochs", default=10, min=1),
                    Field("lr",        "float",  label="lr",     default=0.001,
                          step=0.0001),
                    Field("optimizer", "choice", label="optim",
                          choices=OPTIMIZER_CHOICES, default="adam"),
                    Field("loss",      "choice", label="loss",
                          choices=LOSS_CHOICES, default="crossentropy"),
                    Field("device",    "choice", label="device",
                          choices=device_choices,
                          default=device_default),
                ],
            ),
            ButtonsSection(
                id="controls",
                actions=[
                    Action(id="start", label="Start",
                           rpc="train_start",
                           collect=["datasets", "hyperparams"]),
                    Action(id="pause", label="Pause",  rpc="train_pause"),
                    Action(id="stop",  label="Stop",   rpc="train_stop"),
                ],
            ),
        ],
    )
