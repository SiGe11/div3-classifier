# =============================================================================
# div3.py — train the divisibility-by-3 classifier
# =============================================================================

import math
import time

import numpy as np
import torch
import torch.nn as nn

from model import MAX_DIGITS, Div3Net, encode_batch, ensemble_probs, save_models


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

TRAIN_SIZE = 1_000_000
BATCH = 4096
EPOCHS = 40
LR = 1.5e-3
WEIGHT_DECAY = 1e-5
WARMUP_EPOCHS = 3
SEED = 0

HIDDEN = 256
LAYERS = 3

NUM_MODELS = 2

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Device: {DEVICE}")


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

def generate(n, seed):
    """Sample n numbers as digit histograms."""
    rng = np.random.default_rng(seed)

    lengths = rng.integers(1, MAX_DIGITS + 1, size=n)
    digits = rng.integers(0, 10, size=(n, MAX_DIGITS))

    mask = np.arange(MAX_DIGITS)[None, :] >= (MAX_DIGITS - lengths[:, None])

    multi = lengths > 1
    rows = np.arange(n)[multi]
    digits[rows, (MAX_DIGITS - lengths)[multi]] = rng.integers(1, 10, size=rows.size)

    digits = digits * mask
    hist = np.stack([((digits == d) & mask).sum(1) for d in range(10)], 1)
    labels = (digits.sum(1) % 3 == 0)

    return (
        hist.astype(np.float32),
        lengths.astype(np.float32)[:, None],
        labels.astype(np.float32),
    )


def to_device(arrays):
    return tuple(torch.from_numpy(a).to(DEVICE) for a in arrays)


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def make_scheduler(opt):
    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        span = max(1, EPOCHS - 1 - WARMUP_EPOCHS)
        return 0.5 * (1.0 + math.cos(math.pi * (epoch - WARMUP_EPOCHS) / span))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def train_model(seed, index, total, val):
    print(f"\n=== Training model {index + 1}/{total} (seed={seed}) ===")

    hist, length, y = to_device(generate(TRAIN_SIZE, seed=1000 + seed))
    n = hist.size(0)

    torch.manual_seed(seed)
    net = Div3Net(hidden=HIDDEN, layers=LAYERS).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = make_scheduler(opt)
    lossf = nn.BCEWithLogitsLoss()

    for epoch in range(EPOCHS):
        net.train()
        perm = torch.randperm(n, device=DEVICE)
        running = torch.zeros((), device=DEVICE)

        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = lossf(net(hist[idx], length[idx]), y[idx])
            loss.backward()
            opt.step()
            running += loss.detach() * idx.numel()

        sched.step()

        if epoch % 5 == 4 or epoch == EPOCHS - 1:
            va = accuracy([net], *val)
            print(f"  epoch {epoch + 1:2d}/{EPOCHS}  loss {running.item()/n:.5f}"
                  f"  lr {sched.get_last_lr()[0]:.6f}  val {va:.4f}")

    net.eval()
    return net


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

@torch.no_grad()
def accuracy(models, hist, length, y, chunk=65536):
    for m in models:
        m.eval()
    correct = 0
    for i in range(0, len(y), chunk):
        h = torch.from_numpy(hist[i:i + chunk]).to(DEVICE)
        l = torch.from_numpy(length[i:i + chunk]).to(DEVICE)
        p = ensemble_probs(models, h, l)
        pred = (p > 0.5).float().cpu().numpy()
        correct += (pred == y[i:i + chunk]).sum()
    for m in models:
        m.train()
    return correct / len(y)


def exhaustive(limit):
    nums = list(range(limit))
    hist, length = encode_batch(nums)
    y = np.array([float(n % 3 == 0) for n in nums], dtype=np.float32)
    return hist.numpy(), length.numpy(), y


def main():
    start = time.time()

    val = generate(50_000, seed=555)

    models = [
        train_model(SEED + 100 * k, k, NUM_MODELS, val)
        for k in range(NUM_MODELS)
    ]

    train_seconds = time.time() - start
    print(f"\nTraining wall time: {train_seconds:.1f} s")

    print("\n--- accuracy ---")
    print(f"exhaustive 0..9,999 (SEEN in training)  : "
          f"{accuracy(models, *exhaustive(10_000)):.4f}")

    test = generate(100_000, seed=999)
    print(f"unseen random, 1..{MAX_DIGITS} digits        : "
          f"{accuracy(models, *test):.4f}")

    print("\n--- accuracy by digit count (held out) ---")
    hist, length, y = generate(300_000, seed=4242)
    for lo, hi in [(1, 5), (6, 10), (11, 20), (21, 30), (31, 40), (41, 50)]:
        k = (length[:, 0] >= lo) & (length[:, 0] <= hi)
        print(f"{lo:2d}-{hi:2d} digits : {accuracy(models, hist[k], length[k], y[k]):.4f}"
              f"   (n={k.sum():,})")

    print("\n--- samples ---")
    samples = [
        3, 7, 12, 123456, 999999999999, 100000000002, 42, 43, 44,
        99999999999999999999, 11111111111111111111,
        int("9" * 50), int("1" * 50), int("1" + "0" * 48 + "1"),
        int("1234567890" * 5),
    ]
    h, l = encode_batch(samples, device=DEVICE)
    probs = ensemble_probs(models, h, l).cpu().tolist()
    for n, p in zip(samples, probs):
        truth = "divisible" if n % 3 == 0 else "not divisible"
        pred = "divisible" if p > 0.5 else "not divisible"
        flag = "" if (p > 0.5) == (n % 3 == 0) else "   <-- MISS"
        print(f"{n:>50}  p={p:.4f}  says {pred:<14} truth: {truth}{flag}")

    save_models(models, "div3_model.pt")
    print(f"\nSaved {len(models)} model(s) to div3_model.pt "
          f"(hidden={HIDDEN}, layers={LAYERS}, "
          f"{sum(p.numel() for p in models[0].parameters()):,} params each)")


if __name__ == "__main__":
    main()