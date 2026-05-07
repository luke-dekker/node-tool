"""RNN audio language model — pure-RNN CTC speech recognizer.

The "RNN audio LM" Luke wants disambiguates to ASR-style: predict the next
character/word given audio frames. With CTC + a multi-layer bidirectional
LSTM on mel-spectrogram input, no conv front-end and no attention, this is
the Graves 2013 / original DeepSpeech 2014 architecture.

Pipeline (pure-RNN, no conv, no attention):

    Data In (A:audio)    → MelSpec(inline) → BiLSTM × 3 → Linear(vocab) ─┐
    Data In (A:sentence) → CharTokenizer ────────────────────────────────┤
                              ↓                                          │
                          target_lengths                                 │
                              ↓                                          ↓
                                                              LossCompute(ctc)
                                                                      ↓
                                                            Data Out (B:loss)

Dataset: works against any audio + transcript dataset that DatasetNode can
load. Tested against LibriSpeech dev-clean (~330 MB, free, no account needed)
and Mozilla Common Voice (CC0; download from datacollective.mozillafoundation.org).
Set the InputMarker(audio) `path` to either:

    data/librispeech/LibriSpeech/dev-clean        # LibriSpeech
    /path/to/cv-corpus-21.0/en                    # Common Voice extract

The DatasetNode auto-detects the format. Audio comes through as variable-
length tensors and gets padded by AudioPadCollate before the encoder.

Vocabulary: lowercase a-z + space + apostrophe (28 chars + blank = 29 classes).
LibriSpeech transcripts are auto-lowercased on load by `_load_librispeech`.
Common Voice transcripts get lowercased inside CharTokenizer (lowercase=True).
"""
from __future__ import annotations
from core.graph import Graph
from templates._helpers import grid

LABEL = "RNN Audio LM (CTC ASR)"
DESCRIPTION = (
    "Pure-RNN speech recognizer (Graves 2013 / DeepSpeech 1 style). "
    "Mel spectrogram → 3-layer bi-LSTM → CTC over characters. "
    "Trains on LibriSpeech, Common Voice, or any audio+transcript dataset."
)

# Mel + LSTM hyperparams — small enough to overfit dev-clean on CPU as a
# sanity check, scale up for real training on LibriSpeech 100h+ later.
_SAMPLE_RATE   = 16000
_N_MELS        = 80
_N_FFT         = 400
_HOP_LENGTH    = 160      # 10 ms @ 16 kHz — pred-time hop
_HIDDEN        = 256
_NUM_LAYERS    = 3
_VOCAB_SIZE    = 29       # 28 chars + 1 blank
_VOCAB         = "abcdefghijklmnopqrstuvwxyz '"


def build(graph: Graph) -> dict[str, tuple[int, int]]:
    from nodes.pytorch.input_marker        import InputMarkerNode
    from nodes.pytorch.audio_pad_collate   import AudioPadCollateNode
    from nodes.pytorch.mel_spectrogram     import MelSpectrogramTransformNode
    from nodes.pytorch.tensor_reshape      import TensorReshapeNode
    from nodes.pytorch.layer               import LayerNode
    from nodes.pytorch.recurrent_layer     import RecurrentLayerNode
    from nodes.pytorch.char_tokenizer      import CharTokenizerNode
    from nodes.pytorch.loss_compute        import LossComputeNode
    from nodes.pytorch.train_marker        import TrainMarkerNode

    pos = grid(step_x=240, step_y=160)
    positions: dict[str, tuple[int, int]] = {}

    # ── Input markers ──────────────────────────────────────────────────────
    # Audio comes in as a list-of-1D-tensors (variable-length clips); the
    # DatasetNode emits the column called `path` for both LibriSpeech and
    # Common Voice. The InputMarker forwards whatever the dataset injected.
    audio_in = InputMarkerNode()
    audio_in.inputs["modality"].default_value = "path"
    audio_in.inputs["path"].default_value     = "data/librispeech/LibriSpeech/dev-clean"
    audio_in.inputs["batch_size"].default_value = 4
    graph.add_node(audio_in); positions[audio_in.id] = pos(col=0, row=0)

    sent_in = InputMarkerNode()
    sent_in.inputs["modality"].default_value = "sentence"
    graph.add_node(sent_in); positions[sent_in.id] = pos(col=0, row=2)

    # ── Audio path: pad → mel-spectrogram → encoder ─────────────────────────
    pad = AudioPadCollateNode()
    pad.inputs["frames_per_sample"].default_value = _HOP_LENGTH
    graph.add_node(pad); positions[pad.id] = pos(col=1, row=0)

    mel = MelSpectrogramTransformNode()
    mel.inputs["sample_rate"].default_value = _SAMPLE_RATE
    mel.inputs["n_mels"].default_value      = _N_MELS
    mel.inputs["n_fft"].default_value       = _N_FFT
    mel.inputs["hop_length"].default_value  = _HOP_LENGTH   # match AudioPadCollate
    graph.add_node(mel); positions[mel.id] = pos(col=2, row=0)

    # MelSpectrogram emits (B, n_mels, T) — transpose to (B, T, n_mels) so
    # the LSTM's batch_first=True / time-major interpretation lines up.
    mel_t = TensorReshapeNode()
    mel_t.inputs["op"].default_value     = "transpose"
    mel_t.inputs["dim"].default_value    = 1
    mel_t.inputs["dim_b"].default_value  = 2
    graph.add_node(mel_t); positions[mel_t.id] = pos(col=3, row=0)

    # 3-layer bidirectional LSTM (Graves 2013). Output dim = HIDDEN * 2.
    lstm = RecurrentLayerNode()
    lstm.inputs["kind"].default_value          = "lstm"
    lstm.inputs["hidden_size"].default_value   = _HIDDEN
    lstm.inputs["num_layers"].default_value    = _NUM_LAYERS
    lstm.inputs["bidirectional"].default_value = True
    lstm.inputs["batch_first"].default_value   = True
    graph.add_node(lstm); positions[lstm.id] = pos(col=4, row=0)

    # Project bi-LSTM output to vocab+blank logits — CTC will log-softmax.
    head = LayerNode()
    head.inputs["kind"].default_value         = "linear"
    head.inputs["out_features"].default_value = _VOCAB_SIZE
    graph.add_node(head); positions[head.id] = pos(col=5, row=0)

    # ── Text path: char-tokenize → CTC targets ──────────────────────────────
    tok = CharTokenizerNode()
    tok.inputs["vocab"].default_value     = _VOCAB
    tok.inputs["lowercase"].default_value = True
    tok.inputs["blank_idx"].default_value = 0
    graph.add_node(tok); positions[tok.id] = pos(col=4, row=2)

    # ── CTC loss ────────────────────────────────────────────────────────────
    # Pred shape from the head is (B, T, vocab); LossCompute auto-transposes
    # to (T, B, vocab) when input_lengths.shape[0] == B.
    loss = LossComputeNode()
    loss.inputs["loss_type"].default_value     = "ctc"
    loss.inputs["blank"].default_value         = 0
    loss.inputs["zero_infinity"].default_value = True
    graph.add_node(loss); positions[loss.id] = pos(col=6, row=1)

    data_out = TrainMarkerNode()
    data_out.inputs["kind"].default_value      = "loss"
    data_out.inputs["task_name"].default_value = "asr_ctc"
    graph.add_node(data_out); positions[data_out.id] = pos(col=7, row=1)

    # ── Wires ───────────────────────────────────────────────────────────────
    # Audio path
    graph.add_connection(audio_in.id, "tensor",     pad.id,    "audio")
    graph.add_connection(pad.id,      "padded",     mel.id,    "waveform")
    graph.add_connection(mel.id,      "spectrogram", mel_t.id,  "t1")
    graph.add_connection(mel_t.id,    "tensor",     lstm.id,   "input_seq")
    graph.add_connection(lstm.id,     "output",     head.id,   "tensor_in")

    # Text path
    graph.add_connection(sent_in.id,  "tensor",     tok.id,    "texts")

    # CTC inputs: pred + target + both lengths
    graph.add_connection(head.id,     "tensor_out",     loss.id, "pred")
    graph.add_connection(tok.id,      "targets",        loss.id, "target")
    graph.add_connection(pad.id,      "lengths",        loss.id, "input_lengths")
    graph.add_connection(tok.id,      "target_lengths", loss.id, "target_lengths")

    # Train marker
    graph.add_connection(loss.id,     "loss",       data_out.id, "tensor_in")
    return positions
