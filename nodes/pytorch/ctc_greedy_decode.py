"""CTC greedy decoder — turn (B, T, C) log-probs into list[str] predictions.

Greedy CTC decode = argmax along C, then collapse runs of consecutive
identical labels into one, then drop blanks. This is the "what does the
model think it heard" probe Luke uses to watch ASR training progress.

The node is a passthrough on the loss path: wire `pred_in` from the
encoder head's output, and `pred_out` carries the same tensor unchanged
into LossCompute. That keeps it on the GraphAsModule active cone so
it runs every training step (not just on graph.execute() inspection).

Per-step prediction lines go to stdout (i.e. the terminal you started
`python launch_web.py` in), throttled to every `print_every` calls.
The first sample of each batch is shown as `target → predicted`; if
`targets` isn't wired only the predicted text shows.
"""
from __future__ import annotations
from core.node import BaseNode, PortType


_DEFAULT_VOCAB = "abcdefghijklmnopqrstuvwxyz '"


def _greedy_decode(logits, vocab: str, blank_idx: int = 0) -> list[str]:
    """logits: (B, T, C) tensor. Returns one decoded string per sample.

    CTC collapse: argmax along C, then for each row drop consecutive
    duplicates, then drop blanks. Index 0 = blank by convention.
    """
    import torch
    if logits is None:
        return []
    if logits.dim() == 3:
        # Pred can come in (B, T, C) OR (T, B, C). Heuristic from the loss
        # node: the smaller dim is usually batch; flip if needed.
        if logits.shape[0] > logits.shape[1] and logits.shape[1] < 64:
            # Looks like (T, B, C) — transpose to (B, T, C)
            logits = logits.transpose(0, 1).contiguous()
    elif logits.dim() == 2:
        # (T, C) → single sample
        logits = logits.unsqueeze(0)
    else:
        return []

    # idx2char map: index 0 stays blank (skipped), 1..N → vocab chars
    i2c = {i + 1: ch for i, ch in enumerate(vocab)}
    out: list[str] = []
    with torch.no_grad():
        ids = logits.argmax(dim=-1)   # (B, T)
        for row in ids.tolist():
            collapsed: list[int] = []
            prev = None
            for v in row:
                if v != prev:
                    collapsed.append(v)
                    prev = v
            # Drop blanks; map remaining indices to chars
            decoded = "".join(i2c.get(v, "") for v in collapsed if v != blank_idx)
            out.append(decoded)
    return out


def _decode_targets(targets, target_lengths, vocab: str) -> list[str]:
    """Reverse the CharTokenizer encoding for display."""
    if targets is None:
        return []
    i2c = {i + 1: ch for i, ch in enumerate(vocab)}
    out: list[str] = []
    rows = targets.tolist() if hasattr(targets, "tolist") else list(targets)
    lens = target_lengths.tolist() if hasattr(target_lengths, "tolist") else None
    for i, row in enumerate(rows):
        n = lens[i] if lens else len(row)
        out.append("".join(i2c.get(v, "") for v in row[:n]))
    return out


class CtcGreedyDecodeNode(BaseNode):
    type_name   = "pt_ctc_greedy_decode"
    label       = "CTC Greedy Decode"
    category    = "Training"
    subcategory = "Loss"
    description = (
        "Greedy-decode CTC log-probs into text. Sits between the encoder "
        "head and CTC loss as a passthrough — wire `pred_in` from the head, "
        "`pred_out` into LossCompute. Optionally wire `targets` + "
        "`target_lengths` to print 'target → predicted' every `print_every` "
        "training steps. Output goes to stdout (the terminal you launched "
        "the server from) — throttled so a long training run doesn't flood."
    )

    def __init__(self):
        self._step = 0
        self._last_decoded: list[str] = []
        self._last_targets: list[str] = []
        super().__init__()

    def _setup_ports(self):
        self.add_input("pred_in",        PortType.TENSOR, default=None,
                       description="(B, T, C) or (T, B, C) logits / log-probs from the head.")
        self.add_input("targets",        PortType.TENSOR, default=None, optional=True,
                       description="(B, S) integer targets from CharTokenizer (for display).")
        self.add_input("target_lengths", PortType.TENSOR, default=None, optional=True,
                       description="(B,) target sequence lengths from CharTokenizer.")
        self.add_input("vocab",          PortType.STRING, default=_DEFAULT_VOCAB,
                       description="Same vocab string used by CharTokenizer.")
        self.add_input("blank_idx",      PortType.INT,    default=0)
        self.add_input("print_every",    PortType.INT,    default=25,
                       description="Print 'target → predicted' every N steps. 0 = never print.")
        self.add_output("pred_out",      PortType.TENSOR,
                        description="Passthrough of `pred_in` — wire into LossCompute.")
        self.add_output("decoded_text",  PortType.ANY,
                        description="list[str] of one decoded string per batch sample.")
        self.add_output("info",          PortType.STRING,
                        description="Last 'target → predicted' formatted preview.")

    def execute(self, inputs):
        pred = inputs.get("pred_in")
        if pred is None:
            return {"pred_out": None, "decoded_text": [],
                    "info": "no pred"}
        vocab = str(inputs.get("vocab") or _DEFAULT_VOCAB)
        blank = int(inputs.get("blank_idx") or 0)
        every = max(0, int(inputs.get("print_every") or 0))

        decoded = _greedy_decode(pred, vocab, blank_idx=blank)
        self._last_decoded = decoded

        targets = inputs.get("targets")
        target_lengths = inputs.get("target_lengths")
        truth: list[str] = []
        if targets is not None:
            truth = _decode_targets(targets, target_lengths, vocab)
            self._last_targets = truth

        # Throttled stdout — avoid flooding training with 84 lines/epoch.
        self._step += 1
        info = ""
        if decoded:
            if truth and len(truth) == len(decoded):
                preview = f"  truth: {truth[0]!r}\n  pred:  {decoded[0]!r}"
            else:
                preview = f"  pred:  {decoded[0]!r}"
            info = preview
            if every > 0 and self._step % every == 0:
                # Print to stdout (the launch_web.py terminal) so the user
                # can watch the model converge in real time.
                print(f"[ctc decode step {self._step}]\n{preview}", flush=True)
        return {"pred_out": pred, "decoded_text": decoded, "info": info}

    def export(self, iv, ov):
        # No training loop in the exported script — just do the decode once.
        pred_in = iv.get("pred_in") or "None"
        out_pred = ov.get("pred_out", "_pred")
        out_text = ov.get("decoded_text", "_text")
        vocab = str(self.inputs["vocab"].default_value or _DEFAULT_VOCAB)
        blank = int(self.inputs["blank_idx"].default_value or 0)
        return ["import torch"], [
            f"_p = {pred_in}",
            f"if _p.dim() == 3 and _p.shape[0] > _p.shape[1] and _p.shape[1] < 64:",
            f"    _p = _p.transpose(0, 1).contiguous()",
            f"_ids = _p.argmax(dim=-1).tolist()",
            f"_vocab = {vocab!r}",
            f"_i2c = {{i + 1: ch for i, ch in enumerate(_vocab)}}",
            f"{out_text} = []",
            f"for _row in _ids:",
            f"    _coll = []",
            f"    _prev = None",
            f"    for _v in _row:",
            f"        if _v != _prev: _coll.append(_v); _prev = _v",
            f"    {out_text}.append(''.join(_i2c.get(v, '') for v in _coll if v != {blank}))",
            f"{out_pred} = {pred_in}",
        ]
