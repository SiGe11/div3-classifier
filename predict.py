# =============================================================================
# predict.py — load the trained model(s) and check divisibility by 3
# -----------------------------------------------------------------------------
# Usage:
#     python3 predict.py 123456
#     python3 predict.py 3 42 999999999999 44
#     python3 predict.py            # interactive mode
# =============================================================================

import sys

import torch

from model import MAX_DIGITS, encode_batch, ensemble_probs, load_models


def predict_one(models, n):
    h, l = encode_batch([n])
    device = next(models[0].parameters()).device
    h, l = h.to(device), l.to(device)
    prob = ensemble_probs(models, h, l).item()
    truth = (n % 3 == 0)
    return {
        "number": n,
        "prob": prob,
        "model_says": prob > 0.5,
        "truth": truth,
        "correct": (prob > 0.5) == truth,
    }


def print_header(r):
    print(f"\nNumber: {r['number']}")


def print_result(r, indent="  "):
    model_label = "DIVISIBLE by 3" if r["model_says"] else "NOT divisible by 3"
    print(f"{indent}model: {model_label}  (p={r['prob']:.4f})")
    truth_label = "divisible by 3" if r["truth"] else "not divisible by 3"
    marker = "OK" if r["correct"] else "MISS — model is wrong here"
    print(f"{indent}check: {truth_label}  [{marker}]")


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    models, cfg = load_models("div3_model.pt", device=device)

    args = sys.argv[1:]

    if not args:
        print(f"Loaded {len(models)} model(s), "
              f"{cfg['MAX_DIGITS']} digits max. "
              "Type a number and press Enter. Ctrl+D to quit.")
        try:
            while True:
                line = input("number> ").strip()
                if not line:
                    continue
                try:
                    n = int(line)
                except ValueError:
                    print("  not a whole number — try again")
                    continue
                try:
                    r = predict_one(models, n)
                except ValueError as e:
                    print(f"  {e}")
                    continue
                print_header(r)
                print_result(r)
        except EOFError:
            print()
        return

    correct_count = 0
    total_count = 0

    for a in args:
        try:
            n = int(a)
        except ValueError:
            print(f"skipping '{a}': not a whole number")
            continue
        try:
            r = predict_one(models, n)
        except ValueError as e:
            print(f"skipping {a}: {e}")
            continue

        print_header(r)
        print_result(r)
        total_count += 1
        if r["correct"]:
            correct_count += 1

    if total_count:
        pct = 100.0 * correct_count / total_count
        print(f"\n{correct_count}/{total_count} correct ({pct:.1f}%) "
              f"— check computed as n % 3 == 0")


if __name__ == "__main__":
    main()