"""Signed geometry checks, also called by check.py."""

import math

import numpy as np
import torch

from geometry import StereographicNormal, mobius_add, radial_inverse, radial_map
from model import UniversalVAE, VAE
from train import elbo_loss


def check_universal():
    dtype = torch.float64
    for dimension in (1, 2, 4):
        for k in (-1.0, -1e-6, 0.0, 1e-6, 1.0):
            curvature = torch.tensor(k, dtype=dtype)
            vector = torch.linspace(0.2, 0.5, dimension, dtype=dtype)
            center = radial_map(torch.full_like(vector, 0.3), curvature)
            dist = StereographicNormal(center, torch.ones_like(vector), curvature)
            chart = radial_map(vector, curvature)
            torch.testing.assert_close(radial_inverse(chart, curvature), vector)
            torch.testing.assert_close(mobius_add(-center, dist.wrap(vector), curvature), chart)
            jacobian = torch.autograd.functional.jacobian(dist.wrap, vector)
            point = dist.wrap(vector)
            log_lambda = -torch.log1p(curvature * point.square().sum() / 4)
            volume = torch.linalg.slogdet(jacobian).logabsdet + dimension * log_lambda
            torch.testing.assert_close(dist.log_prob(point), dist.base.log_prob(vector) - volume)
            samples = dist.rsample((3, 5))
            assert samples.shape == (3, 5, dimension)
            assert torch.isfinite(dist.log_prob(samples)).all()

    # Independently integrate a 2D origin density against manifold volume.
    nodes, weights = np.polynomial.legendre.leggauss(300)
    t = torch.tensor((nodes + 1) / 2, dtype=dtype)
    weights = torch.tensor(weights / 2, dtype=dtype)
    for k in (-1.0, 0.0, 1.0):
        curvature = torch.tensor(k, dtype=dtype)
        if k < 0:
            radius, derivative = 2 * t, torch.full_like(t, 2)
        else:
            radius, derivative = t / (1 - t), 1 / (1 - t).square()
        point = torch.stack((radius, torch.zeros_like(radius)), -1)
        dist = StereographicNormal(torch.zeros(2, dtype=dtype), torch.ones(2, dtype=dtype), curvature)
        metric_volume = (1 + curvature * radius.square() / 4).pow(-2)
        mass = (weights * dist.log_prob(point).exp() * 2 * math.pi * radius * metric_volume * derivative).sum()
        torch.testing.assert_close(mass, torch.tensor(1., dtype=dtype), atol=1e-7, rtol=1e-7)

    # Check pathwise curvature derivatives, including the two-sided limit at zero.
    for k in (-1.0, -1e-6, 0.0, 1e-6, 1.0):
        torch.manual_seed(7)
        model = UniversalVAE(encoder_width=8, decoder_width=8, curvature=k).double()
        x = torch.randn(7, 3, dtype=dtype)
        def objective():
            torch.manual_seed(13)
            return elbo_loss(model, x)
        objective().backward()
        gradient = model.raw_curvature.grad.clone()
        assert torch.isfinite(gradient) and gradient.abs() > 1e-8
        with torch.no_grad():
            model.raw_curvature.fill_(k - 1e-7)
            minus = objective()
            model.raw_curvature.fill_(k + 1e-7)
            plus = objective()
            model.raw_curvature.fill_(k)
        torch.testing.assert_close(gradient, (plus - minus) / 2e-7, rtol=2e-4, atol=1e-5)
        restored = UniversalVAE(encoder_width=8, decoder_width=8).double()
        restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(restored.curvature, model.curvature)

    # At zero, sampling, densities and observations recover the flat baseline.
    torch.manual_seed(3)
    flat = VAE(encoder_width=8, decoder_width=8).double()
    universal = UniversalVAE(encoder_width=8, decoder_width=8).double()
    universal.load_state_dict(flat.state_dict(), strict=False)
    x = torch.randn(8, 3, dtype=dtype)
    torch.manual_seed(4)
    z = flat.posterior(x).rsample()
    torch.manual_seed(4)
    signed_z = universal.posterior(x).rsample()
    torch.testing.assert_close(z, signed_z)
    torch.testing.assert_close(flat.posterior(x).log_prob(z), universal.posterior(x).log_prob(z))
    torch.testing.assert_close(flat.observation(z).mean, universal.observation(z).mean)

    # A real ELBO optimizer step can cross the sign boundary in either direction.
    directions = set()
    for seed in range(20):
        torch.manual_seed(seed)
        model = UniversalVAE(encoder_width=8, decoder_width=8).double()
        x = torch.randn(7, 3, dtype=dtype)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        torch.manual_seed(13)
        elbo_loss(model, x).backward()
        gradient = model.raw_curvature.grad.item()
        start = 1e-5 if gradient > 0 else -1e-5
        with torch.no_grad():
            model.raw_curvature.fill_(start)
        optimizer.zero_grad()
        torch.manual_seed(13)
        elbo_loss(model, x).backward()
        optimizer.step()
        assert start * model.curvature.item() < 0
        directions.add(1 if start > 0 else -1)
        if len(directions) == 2:
            break
    assert directions == {-1, 1}


if __name__ == '__main__':
    check_universal()
    print('Signed curvature geometry, normalization, gradients and sign crossing passed.')
