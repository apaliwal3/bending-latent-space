"""Held-out metrics in nats per 3D point, including Gaussian constants."""

import math

import numpy as np
import torch


@torch.no_grad()
def importance_log_likelihood(
    model, x: torch.Tensor, importance_samples: int = 256, sample_chunk_size: int = 64
) -> torch.Tensor:
    if importance_samples < 1 or sample_chunk_size < 1:
        raise ValueError("importance_samples and sample_chunk_size must be positive")
    q = model.posterior(x)
    total = torch.full((len(x),), -torch.inf, dtype=torch.float64)
    for start in range(0, importance_samples, sample_chunk_size):
        count = min(sample_chunk_size, importance_samples - start)
        z = q.sample((count,))
        # CPU float64 accumulation also supports MPS, which has no float64 tensors.
        weights = model.observation(z).log_prob(x) + model.prior().log_prob(z) - q.log_prob(z)
        weights = weights.to(device="cpu", dtype=torch.float64)
        total = torch.logaddexp(total, torch.logsumexp(weights, dim=0))
    return total - math.log(importance_samples)  # log(mean(exp(log weights))).


@torch.no_grad()
def evaluate(
    model, points: np.ndarray, importance_samples: int = 256,
    sample_chunk_size: int = 64, batch_size: int = 256,
    seed: int = 1234, device: str = "cpu",
) -> tuple[dict, dict]:
    torch.manual_seed(seed)
    model.eval()
    iwll, sampled_elbo, mse = [], [], []
    reconstruction_nll, kl = [], []
    if batch_size < 1 or len(points) < 1:
        raise ValueError("evaluation requires a positive batch size and nonempty points")
    for start in range(0, len(points), batch_size):
        x = torch.as_tensor(points[start : start + batch_size], device=device)
        iwll.append(importance_log_likelihood(
            model, x, importance_samples, sample_chunk_size
        ).cpu().numpy())
        q = model.posterior(x)
        z = q.sample()
        observation = model.observation(z)
        nll = -observation.log_prob(x)
        kl_sample = q.log_prob(z) - model.prior().log_prob(z)
        # Sampled latent, observation mean: works without a manifold posterior mean.
        mse.append((observation.mean - x).square().mean(dim=-1).cpu().numpy())
        reconstruction_nll.append(nll.cpu().numpy())
        kl.append(kl_sample.cpu().numpy())
        sampled_elbo.append((-nll - kl_sample).cpu().numpy())
    per_point = {
        "iwll": np.concatenate(iwll),
        "elbo": np.concatenate(sampled_elbo),
        "reconstruction_mse": np.concatenate(mse),
        "reconstruction_nll": np.concatenate(reconstruction_nll),
        "kl": np.concatenate(kl),
    }
    metrics = {name: float(values.mean()) for name, values in per_point.items()}
    metrics.update({
        "n_points": len(points),
        "importance_samples": importance_samples,
        "iwll_point_std": float(per_point["iwll"].std(ddof=1)) if len(points) > 1 else None,
    })
    return metrics, per_point
