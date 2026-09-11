---
license: mit
library_name: pytorch
tags:
- divisibility
- arithmetic
- custom-architecture
- digit-histogram
---

# Div3Net — Divisibility-by-3 Classifier

A custom PyTorch classifier that predicts whether a non-negative
integer is divisible by 3. Uses a digit histogram encoding: the input
is a count of each digit (0-9) plus the total digit count, so the
network cannot see digit order — it must rediscover the digit-sum rule.

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

## Test
Note: this is just for testing PyTorch and learning MI, with the help of Claude.
