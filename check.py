"""Data, likelihood and curved geometry checks: uv run python check.py."""

import math
from types import SimpleNamespace

import numpy as np
import torch
from torch.distributions import Independent, MultivariateNormal, Normal

import config
from data import generate_data
from evaluate import importance_log_likelihood
from model import CurvedVAE, VAE, WrappedNormal, exp_map_zero, log_map_zero, lorentz_inner
from train import elbo_loss


def check_data():
    settings = {**config.DATA, "name": "sphere", "n_samples": 60,
                "noise_std": 0.05, "parameters": {"radius": 2.0}}
    data = generate_data(settings)
    np.testing.assert_allclose(np.linalg.norm(data["clean"], axis=1), 2.0, atol=2e-7)
    np.testing.assert_allclose(data["curvature"], 0.25)
    np.testing.assert_array_equal(data["points"], generate_data(settings)["points"])
    indices = np.concatenate([data[f"{split}_idx"] for split in ("train", "val", "test")])
    np.testing.assert_array_equal(np.sort(indices), np.arange(60))
    noisier = generate_data({**settings, "noise_std": 0.2})
    np.testing.assert_array_equal(data["clean"], noisier["clean"])
    np.testing.assert_array_equal(data["test_idx"], noisier["test_idx"])
    for name in ("swiss_roll", "saddle"):
        surface = generate_data({**settings, "name": name, "parameters": {}})
        assert surface["points"].shape == (60, 3)
        if name == "swiss_roll":
            np.testing.assert_array_equal(surface["curvature"], 0)
        else:
            assert np.all(surface["curvature"] < 0) and np.ptp(surface["curvature"]) > 0


def check_likelihood():
    torch.manual_seed(8)
    matrix = torch.tensor([[1.0, 0.2], [0.3, -0.7], [0.4, 1.1]], dtype=torch.float64)
    sigma = 0.3
    covariance = torch.linalg.inv(torch.eye(2, dtype=torch.float64) + matrix.T @ matrix / sigma**2)
    model = SimpleNamespace(
        posterior=lambda x: MultivariateNormal(x @ matrix @ covariance / sigma**2, covariance),
        prior=lambda: Independent(Normal(torch.zeros(2, dtype=torch.float64), 1.0), 1),
        observation=lambda z: Independent(Normal(z @ matrix.T, sigma), 1),
    )
    marginal = MultivariateNormal(torch.zeros(3, dtype=torch.float64),
                                 matrix @ matrix.T + sigma**2 * torch.eye(3, dtype=torch.float64))
    for magnitude in (1.0, 100.0):
        x = torch.randn(5, 3, dtype=torch.float64) * magnitude
        for count in (1, 7, 17):
            estimated = importance_log_likelihood(model, x, count, sample_chunk_size=3)
            torch.testing.assert_close(estimated, marginal.log_prob(x), rtol=1e-10, atol=1e-8)


def check_curved_geometry():
    torch.manual_seed(19)
    for dimension in (1, 2, 4):
        for value in (-0.01, -1.0, -2.0):
            curvature = torch.tensor(value, dtype=torch.float64)
            tangent = torch.randn(3, dimension, dtype=torch.float64) * 0.3
            locations = exp_map_zero(tangent, curvature)
            torch.testing.assert_close(log_map_zero(locations, curvature), tangent)
            scale = torch.full_like(tangent, 0.7)
            q = WrappedNormal(locations, scale, curvature)
            samples = q.rsample((7, 2))
            assert samples.shape == (7, 2, 3, dimension + 1)
            assert q.log_prob(samples).shape == (7, 2, 3)
            torch.testing.assert_close(lorentz_inner(samples, samples),
                                       torch.full((7, 2, 3), 1 / value, dtype=torch.float64))

            # Verify density by independently differentiating the wrapping map.
            single = WrappedNormal(locations[0], scale[0], curvature)
            vector = torch.linspace(0.2, 0.6, dimension, dtype=torch.float64)
            point = single.wrap(vector)
            jacobian = torch.autograd.functional.jacobian(single.wrap, vector)
            metric = torch.eye(dimension + 1, dtype=torch.float64)
            metric[0, 0] = -1
            log_volume = 0.5 * torch.linalg.slogdet(jacobian.T @ metric @ jacobian).logabsdet
            expected = single.base.log_prob(vector) - log_volume
            torch.testing.assert_close(single.log_prob(point), expected, rtol=1e-9, atol=1e-9)

    # At the origin, the radial law is exactly that of a Euclidean Gaussian.
    curvature = torch.tensor(-1.0, dtype=torch.float64)
    origin = exp_map_zero(torch.zeros(2, dtype=torch.float64), curvature)
    q = WrappedNormal(origin, torch.ones(2, dtype=torch.float64), curvature)
    radial = torch.tensor([0.0, 1e-8, 0.5, 2.0], dtype=torch.float64)
    points = exp_map_zero(torch.stack((radial, torch.zeros_like(radial)), -1), curvature)
    log_volume = torch.zeros_like(radial)
    log_volume[1:] = (radial[1:].sinh() / radial[1:]).log()
    expected = -math.log(2 * math.pi) - radial.square() / 2 - log_volume
    torch.testing.assert_close(q.log_prob(points), expected)

    # Both the origin and a nonzero center need finite derivatives.
    for magnitude in (0.0, 0.4):
        center = torch.full((2,), magnitude, dtype=torch.float64, requires_grad=True)
        scale = torch.ones(2, dtype=torch.float64, requires_grad=True)
        location = exp_map_zero(center, curvature)
        distribution = WrappedNormal(location, scale, curvature)
        loss = distribution.log_prob(location) + distribution.log_prob(distribution.rsample((5,))).mean()
        gradients = torch.autograd.grad(loss, (center, scale))
        assert all(torch.isfinite(gradient).all() for gradient in gradients)


def check_curved_training():
    torch.manual_seed(23)
    for value in (-0.01, -0.5, -2.0):
        candidate = CurvedVAE(encoder_width=8, decoder_width=8, curvature=value)
        samples = candidate.prior().sample((10,))
        expected_norm = candidate.curvature.double().reciprocal().expand(10)
        torch.testing.assert_close(lorentz_inner(samples, samples), expected_norm)
        assert torch.isfinite(candidate.prior().log_prob(samples)).all()
    model = CurvedVAE(latent_dim=2, encoder_width=8, decoder_width=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    x = torch.randn(12, 3)
    for _ in range(3):
        optimizer.zero_grad()
        loss = elbo_loss(model, x)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                   for parameter in model.parameters())
        optimizer.step()
    assert torch.isfinite(importance_log_likelihood(model, x, 7, sample_chunk_size=3)).all()
    assert model.observation(model.prior().sample((5,))).mean.shape == (5, 3)

    # The curved posterior approaches the flat posterior as K approaches zero.
    flat = VAE(latent_dim=2, encoder_width=8, decoder_width=8).double()
    curved = CurvedVAE(latent_dim=2, encoder_width=8, decoder_width=8, curvature=-1e-8).double()
    curved.load_state_dict(flat.state_dict(), strict=False)
    x = x.double()
    torch.manual_seed(42)
    euclidean = flat.posterior(x).rsample()
    torch.manual_seed(42)
    hyperbolic = curved.posterior(x).rsample()
    torch.testing.assert_close(log_map_zero(hyperbolic, curved.curvature), euclidean,
                               rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(curved.posterior(x).log_prob(hyperbolic),
                               flat.posterior(x).log_prob(euclidean), rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    check_data()
    check_likelihood()
    check_curved_geometry()
    check_curved_training()
    print("Data, likelihood, curved geometry and training checks passed.")
