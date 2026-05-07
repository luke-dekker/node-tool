"""Character-level tokenizer node for CTC targets.

Takes a list of strings (e.g. the `sentence` column from CommonVoiceDataset)
and emits the integer-encoded targets + per-sample lengths that nn.CTCLoss
needs as the second / fourth arguments.

CTC vocabulary convention used here:
  index 0 = blank token (CTC needs a dedicated blank symbol)
  indices 1..N = the characters in `vocab`, in declared order

Default vocab covers lowercase English ASR (a-z, space, apostrophe) → 28
classes total counting blank. Override `vocab` to extend (e.g. add digits
or punctuation). Unknown characters are dropped silently — a real production
build would surface that on `info`, but for an "is the loss finite" smoke
test, dropping is fine.
"""
from __future__ import annotations
from core.node import BaseNode, PortType


# Lowercase English ASR baseline. Order matters: index = position + 1
# (the +1 leaves index 0 free for CTC's blank token).
_DEFAULT_VOCAB = "abcdefghijklmnopqrstuvwxyz '"


class CharTokenizerNode(BaseNode):
    type_name   = "pt_char_tokenizer"
    label       = "Char Tokenizer (CTC)"
    category    = "Data"
    subcategory = "Text"
    description = (
        "Encode a batch of strings as CTC targets.\n"
        "  texts (list[str])      → targets (LongTensor (B, S_max), padded with `blank_idx`)\n"
        "                         + target_lengths (LongTensor (B,))\n"
        "                         + vocab_size (INT, includes blank)\n"
        "Index 0 is reserved for CTC's blank token; characters in `vocab` map\n"
        "to indices 1..len(vocab). Lowercases input by default. Unknown\n"
        "characters are dropped (not mapped to a special token)."
    )

    def relevant_inputs(self, values):
        return ["vocab", "lowercase", "blank_idx"]

    def _setup_ports(self):
        self.add_input("texts",     PortType.ANY, default=None,
                       description="List of strings (one per batch sample). "
                                   "DatasetNode emits this as `sentence` for Common Voice.")
        self.add_input("vocab",     PortType.STRING, default=_DEFAULT_VOCAB,
                       description="Characters mapped to indices 1..N. Index 0 is blank.")
        self.add_input("lowercase", PortType.BOOL,   default=True,
                       description="Lowercase input text before lookup.")
        self.add_input("blank_idx", PortType.INT,    default=0,
                       description="CTC blank token index. Padding uses this value too.")
        self.add_output("targets",        PortType.TENSOR,
                        description="(B, S_max) LongTensor of character indices, padded with `blank_idx`.")
        self.add_output("target_lengths", PortType.TENSOR,
                        description="(B,) LongTensor of unpadded sequence lengths.")
        self.add_output("vocab_size",     PortType.INT,
                        description="Total class count including blank — wire into the model's output Linear.")
        self.add_output("info",           PortType.STRING)

    def execute(self, inputs):
        import torch
        texts = inputs.get("texts")
        vocab = str(inputs.get("vocab") or _DEFAULT_VOCAB)
        lowercase = bool(inputs.get("lowercase", True))
        blank = int(inputs.get("blank_idx") or 0)

        # Char → index map, leaving `blank` slot empty
        # (chars get indices 1..N regardless of `blank` value, then we just
        # use `blank` as the padding fill).
        c2i = {ch: i + 1 for i, ch in enumerate(vocab)}
        vocab_size = len(vocab) + 1   # +1 for blank

        if texts is None:
            return {"targets": None, "target_lengths": None,
                    "vocab_size": vocab_size, "info": "no input"}
        if isinstance(texts, str):
            texts = [texts]
        if not isinstance(texts, (list, tuple)):
            return {"targets": None, "target_lengths": None,
                    "vocab_size": vocab_size,
                    "info": f"expected list[str], got {type(texts).__name__}"}

        encoded: list[list[int]] = []
        n_dropped = 0
        for txt in texts:
            s = str(txt)
            if lowercase:
                s = s.lower()
            ids: list[int] = []
            for ch in s:
                if ch in c2i:
                    ids.append(c2i[ch])
                else:
                    n_dropped += 1
            encoded.append(ids)

        lengths = [len(seq) for seq in encoded]
        if not lengths or max(lengths) == 0:
            return {"targets": None, "target_lengths": None,
                    "vocab_size": vocab_size,
                    "info": "all-empty after tokenization"}

        S_max = max(lengths)
        B = len(encoded)
        targets = torch.full((B, S_max), blank, dtype=torch.long)
        for i, seq in enumerate(encoded):
            if seq:
                targets[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        target_lens = torch.tensor(lengths, dtype=torch.long)

        info = (f"B={B} vocab_size={vocab_size} (blank={blank}) "
                f"S_max={S_max} dropped_chars={n_dropped}")
        return {"targets": targets, "target_lengths": target_lens,
                "vocab_size": vocab_size, "info": info}

    def export(self, iv, ov):
        vocab = str(self.inputs["vocab"].default_value or _DEFAULT_VOCAB)
        lower = bool(self.inputs["lowercase"].default_value)
        blank = int(self.inputs["blank_idx"].default_value or 0)
        texts_var = iv.get("texts") or "[]"
        return ["import torch"], [
            f"_vocab = {vocab!r}",
            f"_c2i = {{ch: i + 1 for i, ch in enumerate(_vocab)}}",
            f"{ov.get('vocab_size', '_vocab_size')} = len(_vocab) + 1   # +1 for CTC blank",
            (f"_encoded = [[_c2i[c] for c in (s.lower() if {lower} else s) if c in _c2i] "
             f"for s in {texts_var}]"),
            f"_lens = [len(seq) for seq in _encoded]",
            f"_S_max = max(_lens) if _lens else 0",
            f"{ov.get('targets', '_targets')} = torch.full((len(_encoded), _S_max), {blank}, dtype=torch.long)",
            f"for _i, _seq in enumerate(_encoded):",
            f"    if _seq:",
            f"        {ov.get('targets', '_targets')}[_i, :len(_seq)] = torch.tensor(_seq, dtype=torch.long)",
            f"{ov.get('target_lengths', '_target_lengths')} = torch.tensor(_lens, dtype=torch.long)",
        ]
