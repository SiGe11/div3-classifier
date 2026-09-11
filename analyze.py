# =============================================================================
# analyze.py — extract and inspect the mechanism the model learned
# -----------------------------------------------------------------------------
# The model's input is a digit histogram, and
#
#     n = digit_sum (mod 3),    digit_sum = sum(d * counts[d])
#
# so the first Linear layer can compute the digit sum outright. That makes the
# first layer's weights directly readable: column d is exactly what one
# occurrence of digit d contributes to each hidden unit.
#
# There are two different weight patterns that both solve the task, because
#     sum(d * counts[d])  ==  sum((d mod 3) * counts[d])   (mod 3)
# Probe 2 asks which one the network actually picked.
# =============================================================================

import numpy as np
import torch

from model import MAX_DIGITS, ensemble_probs, load_models

CLASSES = [{0, 3, 6, 9}, {1, 4, 7}, {2, 5, 8}]


def hist_of(seqs):
    """Build (hist, length) tensors from lists of digits (not from integers)."""
    h = np.zeros((len(seqs), 10), dtype=np.float32)
    l = np.zeros((len(seqs), 1), dtype=np.float32)
    for i, s in enumerate(seqs):
        for d in s:
            h[i, d] += 1.0
        l[i, 0] = len(s)
    return torch.from_numpy(h), torch.from_numpy(l)


def within_across(sim):
    same, diff = [], []
    for i in range(10):
        for j in range(i + 1, 10):
            in_same = any(i in g and j in g for g in CLASSES)
            (same if in_same else diff).append(sim[i, j])
    return float(np.mean(same)), float(np.mean(diff))


models, cfg = load_models("div3_model.pt", device="cpu")
print(f"Model: {cfg['num_models']} x (hidden={cfg['hidden']}, layers={cfg['layers']}), "
      f"MAX_DIGITS={cfg['MAX_DIGITS']}, encoding={cfg['encoding']}\n")

net = models[0]
W = net.net[0].weight.detach().numpy()       # (hidden, 11)
Wd = W[:, :10]                               # per-digit columns
print(f"Inspecting model 1 of {len(models)}; first layer weight {W.shape}\n")


# =============================================================================
# PROBE 1 — Are digits in the same mod-3 class interchangeable to the model?
# =============================================================================
# If the model learned the rule, swapping a 1 for a 4 must not change anything,
# so columns 1 and 4 should coincide. Same-class similarity should be ~1.0.

print("=" * 70)
print("PROBE 1 — First-layer weight columns, by digit")
print("=" * 70)

norms = np.linalg.norm(Wd, axis=0, keepdims=True)
normed = Wd / np.maximum(norms, 1e-8)
sim = normed.T @ normed

print("    " + "".join(f"{d:>7d}" for d in range(10)))
for i in range(10):
    print(f"{i:>3d} " + "".join(f"{sim[i, j]:>7.2f}" for j in range(10)))

same, diff = within_across(sim)
print(f"\nMean similarity WITHIN a mod-3 class:     {same:.3f}")
print(f"Mean similarity ACROSS different classes: {diff:.3f}")
print(f"Separation (within - across):             {same - diff:.3f}")


# =============================================================================
# PROBE 2 — Digit-value weights, or residue-class weights?
# =============================================================================
# Fit each hidden unit's 10 weights two ways and compare explained variance:
#   (a) w[d] ~ alpha*d + beta          "counts the digit sum"   (2 params)
#   (b) w[d] ~ gamma[d mod 3]          "counts residues only"   (3 params)

print("\n" + "=" * 70)
print("PROBE 2 — Which rule do the weights encode?")
print("=" * 70)

d_vals = np.arange(10, dtype=np.float64)
res = d_vals % 3
A_lin = np.stack([d_vals, np.ones(10)], 1)
A_res = np.stack([(res == k).astype(np.float64) for k in range(3)], 1)


def r2(design, w):
    coef, *_ = np.linalg.lstsq(design, w, rcond=None)
    resid = w - design @ coef
    denom = ((w - w.mean()) ** 2).sum()
    return 1.0 - resid @ resid / max(denom, 1e-12)


lin = np.array([r2(A_lin, Wd[h].astype(np.float64)) for h in range(Wd.shape[0])])
rsd = np.array([r2(A_res, Wd[h].astype(np.float64)) for h in range(Wd.shape[0])])

print(f"mean R^2, linear-in-digit-value  (2 params): {lin.mean():.3f}")
print(f"mean R^2, residue-class-only     (3 params): {rsd.mean():.3f}")
print(f"hidden units better fit by residue classes : "
      f"{(rsd > lin).sum()}/{len(lin)}")

strength = np.abs(Wd).sum(0)
strength = strength / strength.sum()
print("\nShare of total first-layer weight mass per digit:")
print("  " + "  ".join(f"{d}:{strength[d]:.3f}" for d in range(10)))


# =============================================================================
# PROBE 3 — Does the output depend on digit sum mod 3?
# =============================================================================

print("\n" + "=" * 70)
print("PROBE 3 — Output vs (digit sum mod 3)")
print("=" * 70)

rng = np.random.default_rng(42)
lens = rng.integers(1, MAX_DIGITS + 1, size=20_000)
seqs = [list(rng.integers(0, 10, size=L)) for L in lens]
h, l = hist_of(seqs)
probs = ensemble_probs(models, h, l).numpy()
sums = np.array([sum(s) for s in seqs])

for k in range(3):
    m = (sums % 3) == k
    print(f"digit sum mod 3 == {k}:  mean p={probs[m].mean():.4f}  "
          f"std={probs[m].std():.4f}  n={m.sum()}")


# =============================================================================
# PROBE 4 — Same digit sum, different lengths (length invariance)
# =============================================================================

print("\n" + "=" * 70)
print("PROBE 4 — Same digit sum, different lengths")
print("=" * 70)

cases = [
    ([9], "9           (1 digit,  sum 9)"),
    ([3, 3, 3], "333         (3 digits, sum 9)"),
    ([1] * 9, "111111111   (9 digits, sum 9)"),
    ([4, 5], "45          (2 digits, sum 9)"),
    ([7], "7           (1 digit,  sum 7)"),
    ([3, 4], "34          (2 digits, sum 7)"),
    ([1] * 7, "1111111     (7 digits, sum 7)"),
    ([8], "8           (1 digit,  sum 8)"),
    ([4, 4], "44          (2 digits, sum 8)"),
    ([2, 3, 3], "233         (3 digits, sum 8)"),
]
h, l = hist_of([s for s, _ in cases])
for (_, label), p in zip(cases, ensemble_probs(models, h, l).numpy()):
    print(f"  {label}   ->  p={p:.4f}  ({'DIV' if p > 0.5 else 'not'})")


# =============================================================================
# PROBE 5 — Two-digit grid
# =============================================================================
# Note: includes a=0, which is not a real two-digit number. It probes the
# learned function, not the training distribution.

print("\n" + "=" * 70)
print("PROBE 5 — Two-digit inputs, p(divisible)")
print("=" * 70)

pairs = [(a, b) for a in range(10) for b in range(10)]
h, l = hist_of([[a, b] for a, b in pairs])
probs = ensemble_probs(models, h, l).numpy()

print("      " + "".join(f"{b:>8d}" for b in range(10)))
for a in range(10):
    print(f"  a={a} " + "".join(f"{probs[a * 10 + b]:>8.2f}" for b in range(10)))

print()
for k in range(3):
    v = np.array([p for (a, b), p in zip(pairs, probs) if (a + b) % 3 == k])
    print(f"(a+b) mod 3 == {k}:  mean={v.mean():.4f}  "
          f"min={v.min():.4f}  max={v.max():.4f}")


# =============================================================================
# PROBE 6 — Where does it break down?
# =============================================================================
# The mod-3 boundary is periodic in the digit sum, which ranges 0..450. This is
# the probe that explains the accuracy drop on long inputs.

print("\n" + "=" * 70)
print("PROBE 6 — Accuracy vs digit sum")
print("=" * 70)

lens = rng.integers(1, MAX_DIGITS + 1, size=200_000)
seqs = [list(rng.integers(0, 10, size=L)) for L in lens]
h, l = hist_of(seqs)
probs = ensemble_probs(models, h, l).numpy()
sums = np.array([sum(s) for s in seqs])
correct = (probs > 0.5) == (sums % 3 == 0)

print(" digit sum   accuracy       n")
for lo in range(0, 451, 50):
    m = (sums >= lo) & (sums < lo + 50)
    if m.sum():
        print(f"  {lo:>3d}-{lo + 49:>3d}   {correct[m].mean():.4f}   {m.sum():>7,}")
