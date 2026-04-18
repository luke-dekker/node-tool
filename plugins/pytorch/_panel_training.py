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


def build_training_panel_spec() -> PanelSpec:
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
                    Field("batch_size", "int",    label="batch",  default=32, min=1),
                    Field("split",      "choice", label="split",  default="train",
                          choices=["train", "test", "val"]),
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
                          choices=["cpu", "cuda", "cuda:0", "cuda:1"],
                          default="cpu"),
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
