# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-experiment research repo: can a bag-of-digits neural network learn divisibility by 3?
The model never sees digit *order* — its input is the digit histogram — so the only signal
available is the multiset of digits, and it can succeed only by rediscovering the digit-sum rule.
`analyze.py` exists to check whether it actually did, rather than memorizing.

No test suite. Dependencies are whatever is in the system Python (3.9, torch 2.8, numpy 2.0) —
no venv, no requirements file. Device selection is MPS-or-CPU; there is no CUDA path.

## Commands

```bash
python3 div3.py                       # train + evaluate + overwrite div3_model.pt (~3.5 min)
python3 predict.py 12 99 123456       # classify numbers from argv
python3 predict.py                    # no args -> interactive REPL, Ctrl+D to quit
python3 analyze.py                    # interpretability probes on the saved checkpoint
```

Training config is module-level constants at the top of `div3.py` (`TRAIN_SIZE`, `EPOCHS`,
`HIDDEN`, `LAYERS`, `NUM_MODELS`); there is no CLI. Training is GPU-bound with the dataset
resident on device, so raising `TRAIN_SIZE` is roughly linear in time and is usually a better
use of budget than raising `HIDDEN` — see "Where the accuracy actually is" below.

## Architecture: everything is the digit histogram

`model.py` is the single definition of the architecture, the encoding, and the checkpoint format.
**Do not reintroduce a local `Div3Net` in another file.** All three scripts previously carried
their own copy, the copies drifted, and two of the three could no longer load the checkpoint the
third wrote.

A number is encoded as `(counts[10], length)` — no padding, no embedding, no pooling. This works
because:

```
n = digit_sum (mod 3)        and        digit_sum = sum(d * counts[d])
```

so the digit sum is a *linear* function of the histogram and the first `Linear` layer computes it
outright. An earlier version embedded each digit and mean-pooled, handing the head
`pooled = sum(counts[d]*emb[d]) / N` and `N` as separate inputs — which forced it to multiply two
of its own inputs to recover the digit sum, something a ReLU MLP does badly. That, not the mod-3
periodicity, is what its `hidden=1024` four-layer head was mostly spending capacity on. Measured
at 40 epochs x 2 models: 566s / 0.9869 unseen for the old front-end, 43s / 0.9958 for the
histogram one at equal data — 13x faster and more accurate.

Feeding the histogram is not leaking the answer; it is exactly the information mean-pooling
already exposed (the pooled vectors agree to ~5e-7). Feeding the digit *sum* would be leaking it.

## Checkpoint contract

`model.py` owns `save_models` / `load_models`. The config records `hidden`, `layers`,
`num_models` and `encoding`, and `load_models` rejects a `format_version` mismatch with an
explicit "retrain" message. Bump `FORMAT_VERSION` on any architecture change — without the layer
count in the config, a mismatch used to surface as an unreadable shape error inside
`load_state_dict`.

`*.pt` is gitignored: checkpoints are reproducible build artifacts. `div3_model.v1.backup.pt` is
a pre-restructure checkpoint kept only for reference; nothing in the tree can load it.

## Data generation

`div3.py:generate()` samples digit arrays directly in numpy and **never materializes the
integer**. Two facts make this exact, not an approximation:

1. Uniform over `[10^(L-1), 10^L)` is identical to a uniform digit string of length L with a
   nonzero leading digit (single-digit numbers are drawn 0..9).
2. The label depends only on the digit sum.

This replaced ~38.5s of `str()`/bignum work per run with ~0.25s. Keep it that way — the old
`Dataset`/`DataLoader` path cost more in per-sample Python than the model cost in GPU time, which
matters especially on a 2-performance-core machine.

## Evaluation: read the numbers carefully

**`exhaustive 0..9,999` is a memorization check, not generalization.** Because the input is a
histogram, a 400k-sample training run covers 100% of the distinct inputs at 1-3 digits and 98% at
4 digits. Every number in that range has been seen. `div3.py` labels it `(SEEN in training)`;
keep that label.

The real signal is the held-out per-digit-count table. Accuracy degrades with length because the
mod-3 boundary is periodic in a digit sum that reaches 450 at 50 digits — `analyze.py` Probe 6
plots exactly this.

### Where the accuracy actually is

The 41-50 digit bucket is data-limited, not capacity-limited. Going 400k -> 1M samples at the
same architecture moved it 0.9848 -> 0.9915 (error 1.52% -> 0.85%) for 43s -> 202s. Reach for
more data before more parameters.

## Conventions

- Ensemble members are averaged in **probability** space (`ensemble_probs`), never logits, and
  each member draws its **own** independent training sample.
- Labels and "truth" are always computed directly as `n % 3 == 0`; `predict.py` scores the model
  against that on every call, so its output doubles as a live accuracy check.
- Accumulate training loss on-device and sync once per epoch. A per-batch `.item()` forces a GPU
  sync every step.
- `predict.py` runs on CPU by design: for a single 10-element input, MPS setup and transfer cost
  more than the matmuls save.
