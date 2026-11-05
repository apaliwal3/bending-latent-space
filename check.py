"""Two small correctness checks. Run with: uv run python check.py"""

from types import SimpleNamespace

import numpy as np
import torch
from torch.distributions import Independent, MultivariateNormal, Normal

import config
from data import generate_data
from evaluate import importance_log_likelihood


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


if __name__ == "__main__":
    check_data()
    check_likelihood()
    print("Data and likelihood checks passed.")
