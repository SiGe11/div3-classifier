# =============================================================================
# model.py — single definition of the architecture and the checkpoint format
# =============================================================================

import torch
import torch.nn as nn

MAX_DIGITS = 50
FORMAT_VERSION = 4


# -----------------------------------------------------------------------------
# Encoding
# -----------------------------------------------------------------------------

def encode(n: int):
    """Return (histogram of 10 digit counts, number of digits) for n >= 0."""
    if n < 0:
        raise ValueError("This experiment supports only non-negative integers.")
    s = str(n)
    if len(s) > MAX_DIGITS:
        raise ValueError(
            f"number has {len(s)} digits, but MAX_DIGITS={MAX_DIGITS}"
        )
    hist = [0.0] * 10
    for c in s:
        hist[int(c)] += 1.0
    return hist, float(len(s))


def encode_batch(nums, device="cpu"):
    """Encode a sequence of ints into (hist (B,10), length (B,1)) tensors."""
    pairs = [encode(n) for n in nums]
    hist = torch.tensor([h for h, _ in pairs], dtype=torch.float32, device=device)
    length = torch.tensor([[l] for _, l in pairs], dtype=torch.float32, device=device)
    return hist, length


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

class Div3Net(nn.Module):
    def __init__(self, hidden=256, layers=3):
        super().__init__()
        self.hidden = hidden
        self.layers = layers

        mods, d = [], 11          # 10 digit counts + 1 length feature
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        self.net = nn.Sequential(*mods, nn.Linear(d, 1))

    def forward(self, hist, length):
        x = torch.cat([hist, length / MAX_DIGITS], dim=1)
        return self.net(x).squeeze(-1)


# -----------------------------------------------------------------------------
# Checkpoint I/O
# -----------------------------------------------------------------------------

def save_models(models, path="div3_model.pt"):
    torch.save(
        {
            "format_version": FORMAT_VERSION,
            "config": {
                "MAX_DIGITS": MAX_DIGITS,
                "hidden": models[0].hidden,
                "layers": models[0].layers,
                "num_models": len(models),
                "encoding": "digit-histogram",
            },
            "state_dicts": [m.state_dict() for m in models],
        },
        path,
    )


def load_models(path="div3_model.pt", device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    version = ckpt.get("format_version")

    if version != FORMAT_VERSION:
        raise RuntimeError(
            f"{path} is format_version {version}, but this code expects "
            f"{FORMAT_VERSION}. Retrain with `python3 div3.py`."
        )

    cfg = ckpt["config"]
    models = []
    for sd in ckpt["state_dicts"]:
        m = Div3Net(hidden=cfg["hidden"], layers=cfg["layers"])
        m.load_state_dict(sd)
        m.to(device)
        m.eval()
        models.append(m)
    return models, cfg


@torch.no_grad()
def ensemble_probs(models, hist, length):
    """Average the ensemble's probabilities (never logits)."""
    return torch.stack(
        [torch.sigmoid(m(hist, length)) for m in models], dim=0
    ).mean(dim=0)