"""Saved-threshold BatchTopK inference, adapted from dictionary_learning (MIT).

The trained decoder is not needed to read concepts. See NOTICE for attribution.
"""

import torch
from torch import nn


def normalize_embeddings(embeddings):
    """Use the same per-token normalization as SAE training."""
    if not embeddings.is_floating_point():
        embeddings = embeddings.float()
    if not torch.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN or infinite values")
    minimum = max(1e-8, torch.finfo(embeddings.dtype).tiny)
    return embeddings / embeddings.norm(dim=-1, keepdim=True).clamp_min(minimum)


class SAE(nn.Module):
    activation_dim = 768
    dict_size = 1536

    def __init__(self, state):
        super().__init__()
        expected = {"encoder.weight", "encoder.bias", "decoder.weight", "b_dec", "k", "threshold"}
        if set(state) != expected:
            raise ValueError("Expected a saved BatchTopK autoencoder state")
        if (state["encoder.weight"].shape != (1536, 768)
                or state["decoder.weight"].shape != (768, 1536)
                or state["encoder.bias"].shape != (1536,)
                or state["b_dec"].shape != (768,)
                or int(state["k"]) != 8 or float(state["threshold"]) < 0):
            raise ValueError("Expected the released 768/1536/k=8 SAE")
        if any(not torch.isfinite(value).all() for value in state.values()):
            raise ValueError("SAE weights must be finite")
        self.encoder = nn.Linear(768, 1536)
        # Preserve the saved tensor layout as well as its values.
        self.encoder.weight = nn.Parameter(state["encoder.weight"].float().clone(), requires_grad=False)
        self.encoder.bias = nn.Parameter(state["encoder.bias"].float().clone(), requires_grad=False)
        self.b_dec = nn.Parameter(state["b_dec"].float().clone(), requires_grad=False)
        self.register_buffer("threshold", state["threshold"].float().clone())
        self.register_buffer("k", state["k"].clone())

    def encode(self, x):
        activations = torch.relu(self.encoder(x - self.b_dec))
        return activations * (activations > self.threshold)

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        state = torch.load(path, map_location="cpu", weights_only=True)
        return cls(state).to(device).eval()
