"""TextDatasetNode — char-level text dataset with sliding-window batches.

The text equivalent of MNISTDatasetNode for char-level language modeling.
Loads a text file, builds a character vocabulary, and produces a DataLoader
that yields `(input_seq, target_seq)` pairs where target is the input shifted
by one position (next-char prediction).

Output:
    dataloader: yields tuples (x, y) where
        x: LongTensor of shape (batch_size, seq_len)
        y: LongTensor of shape (batch_size, seq_len)
        with y[i, t] == x[i, t+1] (or the next char from the text)

If the path doesn't exist, falls back to a small built-in synthetic corpus
so the template trains out of the box without external data.
"""
from __future__ import annotations
from typing import Any
from core.node import BaseNode, PortType


# Tiny built-in corpus — repeating bigrams so a char-LM converges in seconds.
# Used when the user-supplied path doesn't exist; lets the template demo work
# without anyone having to download tinyshakespeare or similar.
_FALLBACK_CORPUS = (
    "the quick brown fox jumps over the lazy dog. " * 200 +
    "pack my box with five dozen liquor jugs. " * 200 +
    "how vexingly quick daft zebras jump! " * 200
)


class TextDatasetNode(BaseNode):
    type_name   = "pt_text_dataset"
    label       = "Text Dataset"
    category    = "Datasets"
    subcategory = "Loader"
    description = (
        "Char-level text dataset for language modeling. Loads a .txt file, "
        "builds a char vocab, and yields sliding-window (input, target) "
        "pairs where target is input shifted by one. If the path is missing, "
        "falls back to a small built-in corpus so demos work out of the box. "
        "Outputs vocab_size as a separate port — wire it to your Embedding "
        "and final Linear layers to match the vocab dimensions."
    )

    def __init__(self):
        self._cached: dict | None = None
        self._cached_cfg: tuple = ()
        super().__init__()

    def _setup_ports(self) -> None:
        self.add_input("path",       PortType.STRING, default="data/text.txt",
                       description="Path to a .txt file (falls back to built-in if missing)")
        self.add_input("task_id",    PortType.STRING, default="default",
                       description="Pairs this dataset with a Train Output that has the same task_name")
        self.add_input("seq_len",    PortType.INT,    default=64,
                       description="Window length per training sample")
        self.add_input("batch_size", PortType.INT,    default=32)
        self.add_input("shuffle",    PortType.BOOL,   default=True)
        # Tensor preview outputs — wire x → Embedding, label → ReshapeForLoss
        self.add_output("x",          PortType.TENSOR,
                        description="One batch of input sequences (B, seq_len) int64")
        self.add_output("label",      PortType.TENSOR,
                        description="Shifted target sequences (B, seq_len) int64")
        self.add_output("dataloader", PortType.DATALOADER)
        self.add_output("vocab_size", PortType.INT,
                        description="Number of distinct chars — use as Embedding "
                                    "num_embeddings and final Linear out_features")
        self.add_output("info",       PortType.STRING)

    def _build(self, path: str, seq_len: int, batch_size: int, shuffle: bool) -> dict:
        from pathlib import Path
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        # Load corpus (or fall back if missing)
        p = Path(path)
        if p.exists() and p.is_file():
            try:
                text = p.read_text(encoding="utf-8")
                source = str(p)
            except Exception:
                text = _FALLBACK_CORPUS
                source = "built-in fallback"
        else:
            text = _FALLBACK_CORPUS
            source = "built-in fallback"

        # Build char vocab
        chars = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(chars)}
        vocab_size = len(chars)

        # Encode entire text as a flat int tensor
        data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

        # Slide windows: x[i] = data[i:i+seq_len], y[i] = data[i+1:i+seq_len+1]
        n_windows = (len(data) - 1) // seq_len
        if n_windows < 1:
            return {"loader": None, "vocab_size": vocab_size, "info": f"text too short for seq_len={seq_len}"}

        usable = n_windows * seq_len
        x = data[:usable].reshape(n_windows, seq_len)
        y = data[1:usable + 1].reshape(n_windows, seq_len)

        ds = TensorDataset(x, y)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True)
        info = (f"text={source}  chars={len(text):,}  vocab={vocab_size}  "
                f"windows={n_windows:,}  batches={len(loader)}  seq_len={seq_len}")
        return {"loader": loader, "vocab_size": vocab_size, "info": info}

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        path       = str(inputs.get("path") or "data/text.txt")
        seq_len    = max(1, int(inputs.get("seq_len") or 64))
        batch_size = max(1, int(inputs.get("batch_size") or 32))
        shuffle    = bool(inputs.get("shuffle", True))

        cfg = (path, seq_len, batch_size, shuffle)
        if self._cached is None or self._cached_cfg != cfg:
            try:
                self._cached = self._build(path, seq_len, batch_size, shuffle)
                self._cached_cfg = cfg
            except Exception as exc:
                return {"dataloader": None, "vocab_size": 0,
                        "info": f"build failed: {exc}"}

        loader = self._cached["loader"]
        # Sample one batch for tensor preview
        x, label = None, None
        if loader is not None:
            try:
                batch = next(iter(loader))
                x, label = batch[0], batch[1]
            except Exception:
                pass

        return {
            "x":          x,
            "label":      label,
            "dataloader": loader,
            "vocab_size": self._cached["vocab_size"],
            "info":       self._cached["info"],
        }

    def export(self, iv, ov):
        path       = self._val(iv, "path")
        seq_len    = self._val(iv, "seq_len")
        batch_size = self._val(iv, "batch_size")
        shuffle    = self._val(iv, "shuffle")
        dl_var     = ov.get("dataloader", "_text_dl")
        vocab_var  = ov.get("vocab_size", "_text_vocab")
        info_var   = ov.get("info",       "_text_info")
        return [
            "import torch",
            "from torch.utils.data import DataLoader, TensorDataset",
            "from pathlib import Path",
        ], [
            f"_p = Path({path})",
            f"_text = _p.read_text(encoding='utf-8') if _p.exists() else 'hello world ' * 1000",
            f"_chars = sorted(set(_text))",
            f"_stoi  = {{ch: i for i, ch in enumerate(_chars)}}",
            f"{vocab_var} = len(_chars)",
            f"_data  = torch.tensor([_stoi[c] for c in _text], dtype=torch.long)",
            f"_n     = (len(_data) - 1) // {seq_len}",
            f"_usable = _n * {seq_len}",
            f"_x = _data[:_usable].reshape(_n, {seq_len})",
            f"_y = _data[1:_usable + 1].reshape(_n, {seq_len})",
            f"{dl_var} = DataLoader(TensorDataset(_x, _y), batch_size={batch_size}, "
            f"shuffle={shuffle}, drop_last=True)",
            f"{info_var} = f'text={{len(_text)}} chars, vocab={{{vocab_var}}}, "
            f"windows={{_n}}'",
        ]
