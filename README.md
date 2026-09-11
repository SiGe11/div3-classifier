# Div3Net — Divisibility-by-3 Classifier

A custom PyTorch classifier that predicts whether a non-negative integer is divisible by 3. Uses a digit histogram encoding: the input is a count of each digit (0-9) plus the total digit count, so the network cannot see digit order — it must rediscover the digit-sum rule.

## Architecture

- Input: 10 digit counts + 1 normalized digit-count feature
- Body: `layers` hidden layers of `hidden` units, ReLU between them
- Output: single logit
- Params: ~135K per model, ensemble of N models

## Checkpoint format

The file `div3_model.pt` is a dict with:

- `format_version`: 4
- `config`: `{MAX_DIGITS, hidden, layers, num_models, encoding}`
- `state_dicts`: list of `state_dict` objects, one per ensemble member

Load with the `model.py` file from the same project.

## Performance

~99.8% on held-out random numbers spanning 1 to 50 digits.

## How to use

### Requirements

- Python 3.9 or later
- PyTorch 2.x
- NumPy

Install with:

```bash
pip install torch numpy
```

On an Apple Silicon Mac, PyTorch will automatically use the GPU via the `mps` device. No extra install steps are needed.

### Files

| File | Purpose |
|------|---------|
| `model.py`   | Architecture, encoding, checkpoint I/O — imported by the other scripts |
| `div3.py`    | Training script; writes `div3_model.pt` |
| `predict.py` | Loads `div3_model.pt` and classifies numbers from the command line |

Keep all three files in the same folder. `predict.py` will not work without `model.py` next to it.

### Training

Run:

```bash
python3 div3.py
```

This will:

1. Generate 1,000,000 random numbers (1 to 50 digits) as digit histograms.
2. Train an ensemble of 2 models for 40 epochs each.
3. Print a loss curve and validation accuracy every 5 epochs.
4. Evaluate on held-out sets and print per-digit-count accuracy.
5. Save the trained ensemble to `div3_model.pt`.

Expected runtime on an M-series Mac: a few minutes. The loss should drop steadily and the validation accuracy should reach ~99.8%.

You can adjust the configuration at the top of `div3.py`:

- `TRAIN_SIZE` — examples per model
- `EPOCHS` — passes over the training data
- `BATCH` — examples per update
- `LR` — peak learning rate
- `HIDDEN`, `LAYERS` — model size
- `NUM_MODELS` — ensemble size

### Predicting

Once `div3_model.pt` exists, classify numbers from the command line:

```bash
python3 predict.py 123456
python3 predict.py 3 7 42 43 99999999999999999999
```

Each result prints in two lines — first the model's answer, then the mathematical check:

```
Number: 123456
  model: DIVISIBLE by 3  (p=1.0000)
  check: divisible by 3  [OK]
```

Run without arguments to enter interactive mode:

```bash
python3 predict.py
```

```
Loaded 2 model(s), 50 digits max. Type a number and press Enter. Ctrl+D to quit.
number> 42

Number: 42
  model: DIVISIBLE by 3  (p=1.0000)
  check: divisible by 3  [OK]
number>
```

The **model line** shows the ensemble's verdict and its averaged probability. The **check line** shows the ground truth computed as `n % 3 == 0`. When they disagree, a `MISS` marker flags the case.

Numbers with more than `MAX_DIGITS` (50) digits or negative values are rejected with an explanatory message.

## Test

Note: this is just for testing PyTorch and learning ML, with the help of Claude.