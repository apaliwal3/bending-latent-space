"""Held-out metrics in nats per 3D point, including Gaussian constants."""

import math
from copy import deepcopy
import warnings

import numpy as np
import torch


@torch.no_grad()
def importance_log_likelihood(
    model, x: torch.Tensor, importance_samples: int = 256, sample_chunk_size: int = 64,
    prior_fraction: float = 0.1,
) -> torch.Tensor:
    """Stratified posterior/prior mixture estimate; finite samples can still miss mass."""
    return _importance_estimate(model, x, importance_samples, sample_chunk_size, prior_fraction)[0]


@torch.no_grad()
def _importance_estimate(model, x, importance_samples, sample_chunk_size, prior_fraction):
    if importance_samples < 1 or sample_chunk_size < 1:
        raise ValueError("importance_samples and sample_chunk_size must be positive")
    if not 0 <= prior_fraction <= 1:
        raise ValueError("prior_fraction must lie in [0, 1]")
    q, prior = model.posterior(x), model.prior()
    prior_count = round(importance_samples * prior_fraction)
    fraction = prior_count / importance_samples
    total = torch.full((len(x),), -torch.inf, dtype=torch.float64)
    squared_total = total.clone()
    for from_prior, draws in ((False, importance_samples - prior_count), (True, prior_count)):
        for start in range(0, draws, sample_chunk_size):
            count = min(sample_chunk_size, draws - start)
            z = prior.sample((count, len(x))) if from_prior else q.sample((count,))
            log_prior = prior.log_prob(z)
            if fraction == 0:
                proposal = q.log_prob(z)
            elif fraction == 1:
                proposal = log_prior
            else:
                proposal = torch.logaddexp(q.log_prob(z) + math.log1p(-fraction),
                                           log_prior + math.log(fraction))
            # CPU float64 accumulation also supports MPS.
            weights = (model.observation(z).log_prob(x) + log_prior - proposal).to(
                device="cpu", dtype=torch.float64)
            if not torch.isfinite(weights).all():
                raise FloatingPointError("Nonfinite importance weights; check latent geometry.")
            total = torch.logaddexp(total, torch.logsumexp(weights, dim=0))
            squared_total = torch.logaddexp(squared_total, torch.logsumexp(2 * weights, dim=0))
    ess = (2 * total - squared_total).exp().clamp_max(importance_samples)
    return total - math.log(importance_samples), ess


@torch.no_grad()
def quadrature_log_likelihood(model, points, grid_size=400, extent=8., tolerance=.01):
    """CPU midpoint integration for our 2D Gaussian-base priors and Gaussian likelihoods.

    Checks a doubled grid and bounds omitted Gaussian prior tails. This bypasses
    the encoder entirely; passing these checks is numerical evidence, not a proof.
    """
    if model.latent_dim != 2 or grid_size < 2 or extent <= 0 or tolerance <= 0:
        raise ValueError("Quadrature requires latent_dim=2 and positive grid/extent/tolerance.")
    model = deepcopy(model).cpu().eval()
    parameter = next(model.parameters())
    x = torch.as_tensor(points, dtype=torch.float64, device="cpu")
    sigma = float(model.observation_std)
    constant = -1.5 * math.log(2 * math.pi) - 3 * math.log(sigma)
    estimates = []
    for cells in (grid_size, 2 * grid_size):
        step = 2 * extent / cells
        axis = torch.linspace(-extent + step / 2, extent - step / 2, cells, dtype=torch.float64)
        a, b = torch.meshgrid(axis, axis, indexing="ij")
        grid = torch.stack((a, b), -1).reshape(-1, 2)
        total = torch.full((len(x),), -torch.inf, dtype=torch.float64)
        for u in grid.split(4096):
            z = model.latent_from_tangent(u.to(parameter.dtype))
            mean = model.observation(z).mean.double()
            # Matrix distances avoid allocating a grid x points x coordinates tensor.
            squared = (mean.square().sum(-1, keepdim=True) + x.square().sum(-1)[None, :]
                       - 2 * mean @ x.T).clamp_min(0)
            log_prior_mass = (-.5 * u.square().sum(-1) - math.log(2 * math.pi)
                              + 2 * math.log(step))[:, None]
            values = constant - squared / (2 * sigma**2) + log_prior_mass
            total = torch.logaddexp(total, torch.logsumexp(values, dim=0))
        estimates.append(total)
    coarse, fine = estimates
    grid_change = float((fine - coarse).abs().max())
    outside_one_axis = math.erfc(extent / math.sqrt(2))
    outside_square = 2 * outside_one_axis - outside_one_axis**2
    log_tail_bound = constant + math.log(outside_square) if outside_square else -math.inf
    tail_error = float(torch.nn.functional.softplus(log_tail_bound - fine).max())
    if not torch.isfinite(fine).all() or grid_change > tolerance or tail_error > tolerance:
        raise ValueError("Likelihood integration unresolved: "
                         f"grid change={grid_change:.3g}, tail bound={tail_error:.3g} nats. "
                         "Increase quadrature_grid_size or quadrature_extent in evaluation settings.")
    return fine, {"quadrature_grid_change": grid_change, "quadrature_tail_bound": tail_error}


@torch.no_grad()
def evaluate(
    model, points: np.ndarray, importance_samples: int = 256,
    sample_chunk_size: int = 64, batch_size: int = 256,
    seed: int = 1234, device: str = "cpu", prior_fraction: float = 0.1,
    quadrature_grid_size: int = 400, quadrature_extent: float = 8.,
) -> tuple[dict, dict]:
    torch.manual_seed(seed)
    model.eval()
    iwll, importance_ess, sampled_elbo, mse = [], [], [], []
    reconstruction_nll, kl = [], []
    if batch_size < 1 or len(points) < 1:
        raise ValueError("evaluation requires a positive batch size and nonempty points")
    for start in range(0, len(points), batch_size):
        x = torch.as_tensor(points[start : start + batch_size], device=device)
        estimate, ess = _importance_estimate(model, x, importance_samples, sample_chunk_size, prior_fraction)
        iwll.append(estimate.numpy())
        importance_ess.append(ess.numpy())
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
        "importance_ess": np.concatenate(importance_ess),
        "elbo": np.concatenate(sampled_elbo),
        "reconstruction_mse": np.concatenate(mse),
        "reconstruction_nll": np.concatenate(reconstruction_nll),
        "kl": np.concatenate(kl),
    }
    metrics = {name: float(values.mean()) for name, values in per_point.items()}
    metrics.update({
        "n_points": len(points),
        "importance_samples": importance_samples,
        "prior_fraction": round(importance_samples * prior_fraction) / importance_samples,
        "iwll_point_std": float(per_point["iwll"].std(ddof=1)) if len(points) > 1 else None,
        "importance_ess_min": float(per_point["importance_ess"].min()),
        "likelihood_method": "importance_estimate",
        "likelihood_checked": False,
    })
    if model.latent_dim == 2:
        direct, diagnostics = quadrature_log_likelihood(
            model, points, quadrature_grid_size, quadrature_extent)
        per_point["log_likelihood"] = direct.numpy()
        metrics.update(diagnostics, likelihood_method="quadrature", likelihood_checked=True)
        gap = np.abs(per_point["log_likelihood"] - per_point["iwll"])
        metrics["importance_max_gap"] = float(gap.max())
        if gap.max() > .1:
            warnings.warn("Importance estimates disagree with direct integration; "
                          "use log_likelihood for the comparison.", RuntimeWarning, stacklevel=2)
    else:
        per_point["log_likelihood"] = per_point["iwll"].copy()
        if metrics["importance_ess_min"] < max(10, .01 * importance_samples):
            warnings.warn("Low importance effective sample size; likelihood estimates need "
                          "further validation before ranking models.", RuntimeWarning, stacklevel=2)
    metrics["log_likelihood"] = float(per_point["log_likelihood"].mean())
    return metrics, per_point
