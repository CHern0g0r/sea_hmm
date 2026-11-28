import torch
from torch import Tensor


def _normalize_probabilities(values, shape: tuple[int, ...], name: str) -> Tensor:
    probabilities = torch.as_tensor(values, dtype=torch.float64, device="cpu").detach()
    if probabilities.shape != shape or not torch.isfinite(probabilities).all():
        raise ValueError(f"{name} must have shape {shape} and contain finite values")
    totals = probabilities.sum(dim=-1, keepdim=True)
    if (probabilities < 0).any() or not torch.allclose(
        totals, torch.ones_like(totals), rtol=0, atol=1e-6
    ):
        raise ValueError(f"{name} must contain nonnegative probabilities that sum to one")
    return probabilities / totals
