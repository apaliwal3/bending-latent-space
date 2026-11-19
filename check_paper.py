"""Numerical and experiment checks for the paper's mixed-curvature extension."""

import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import warnings

import numpy as np
import torch

import config
from evaluate import evaluate
from geometry import log_at_origin
from mixed import MixedCurvatureVAE, PaperFlatVAE, partition_dimensions
from model import WrappedNormal, exp_map_zero
from paper_geometry import ProductWrappedNormal, ProjectedWrappedNormal, exp_origin
from train import elbo_loss, train


def check_maps_and_jacobians():
    dtype = torch.float64
    for d in (2, 3, 5):
        for k in (-1., -1e-6, 0., 1e-6, 1.):
            curvature = torch.tensor(k, dtype=dtype)
            v = torch.linspace(.1, .4, d, dtype=dtype)
            center = exp_origin(torch.full_like(v, .2), curvature)
            dist = ProjectedWrappedNormal(center, torch.full_like(v, .3), curvature)
            point = dist.wrap(v)
            torch.testing.assert_close(log_at_origin(exp_origin(v, curvature), curvature), v)
            jacobian = torch.autograd.functional.jacobian(dist.wrap, v)
            log_volume = (torch.linalg.slogdet(jacobian).logabsdet
                          - d * torch.log1p(curvature * point.square().sum() / 4))
            torch.testing.assert_close(dist.log_prob(point), dist.base.log_prob(v) - log_volume,
                                       atol=1e-8, rtol=1e-8)
            batched = ProjectedWrappedNormal(center.expand(4, -1), torch.full_like(v, .3), curvature)
            samples = batched.rsample((3, 2))
            assert samples.shape == (3, 2, 4, d)
            assert batched.log_prob(samples).shape == (3, 2, 4)
            assert torch.isfinite(batched.log_prob(samples)).all()

    # Independent Lorentz transport/exp implementation must give identical H samples/densities.
    k = torch.tensor(-.7, dtype=dtype)
    center_vector = torch.tensor([.2, -.1], dtype=dtype)
    scale = torch.tensor([.4, .7], dtype=dtype)
    lorentz = WrappedNormal(exp_map_zero(center_vector, k), scale, k)
    projected = ProjectedWrappedNormal(exp_origin(center_vector, k), scale, k)
    v = torch.tensor([[.3, -.5], [-.7, .6]], dtype=dtype)
    ambient = lorentz.wrap(v)
    expected = 2 * ambient[..., 1:] / (1 + (-k).sqrt() * ambient[..., :1])
    torch.testing.assert_close(projected.wrap(v), expected)
    torch.testing.assert_close(projected.log_prob(expected), lorentz.log_prob(ambient))

    # Independent ambient-sphere transport and exponential, including beyond the cut locus.
    radius = torch.tensor(1.2, dtype=dtype)
    k = radius.pow(-2)
    center = exp_origin(center_vector, k)
    denominator = 1 + k * center.square().sum() / 4
    mu = torch.cat(((radius * (1 - k * center.square().sum() / 4) / denominator).reshape(1),
                     center / denominator))
    origin = torch.tensor([radius, 0., 0.], dtype=dtype)
    for v in (torch.tensor([.3, -.5], dtype=dtype), torch.tensor([9., 1.], dtype=dtype)):
        tangent = torch.cat((torch.zeros(1, dtype=dtype), v))
        transported = tangent - (mu * tangent).sum() / (radius * (radius + mu[0])) * (origin + mu)
        r = v.norm()
        ambient = (r / radius).cos() * mu + radius * (r / radius).sin() / r * transported
        expected = 2 * radius * ambient[1:] / (radius + ambient[0])
        dist = ProjectedWrappedNormal(center, scale, k)
        torch.testing.assert_close(dist.wrap(v), expected)


def check_normalization_and_windings():
    dtype = torch.float64
    nodes, weights = np.polynomial.legendre.leggauss(350)
    nodes, weights = torch.tensor(nodes, dtype=dtype), torch.tensor(weights, dtype=dtype)
    for d in (2, 3):
        unit_area = 2 * math.pi ** (d / 2) / math.gamma(d / 2)
        for k in (-1., 0., .1, 1., 4.):
            for sigma in (.3, 1., 3.) if k > 0 else (1.,):
                curvature = torch.tensor(k, dtype=dtype)
                dist = ProjectedWrappedNormal(torch.zeros(d, dtype=dtype),
                                               torch.full((d,), sigma, dtype=dtype), curvature)
                if k > 0:
                    theta = (nodes + 1) * math.pi / 2
                    radial = 2 / math.sqrt(k) * (theta / 2).tan()
                    volume = (unit_area * (theta.sin() / math.sqrt(k)).pow(d - 1)
                              / math.sqrt(k) * math.pi / 2)
                else:
                    r = (nodes + 1) * 6
                    radial = r if k == 0 else 2 * (r / 2).tanh()
                    volume = unit_area * (r if k == 0 else r.sinh()).pow(d - 1) * 6
                points = torch.zeros(len(nodes), d, dtype=dtype)
                points[:, 0] = radial
                mass = (weights * volume * dist.log_prob(points).exp()).sum()
                torch.testing.assert_close(mass, torch.tensor(1., dtype=dtype), atol=2e-8, rtol=2e-8)

    # At high spread the principal inverse misses substantial mass. Compare a long
    # independent sum in dimension 2 against our adaptively bounded winding sum.
    k = torch.tensor(2., dtype=dtype)
    dist = ProjectedWrappedNormal(torch.zeros(2, dtype=dtype), torch.tensor([2., 3.], dtype=dtype), k)
    v = torch.tensor([.2, .4], dtype=dtype)
    r = v.norm()
    values = []
    for n in range(-80, 81):
        signed = r + n * 2 * math.pi / k.sqrt()
        preimage = v * signed / r
        jac = ((signed * k.sqrt()).sin().abs() / (signed * k.sqrt()).abs()).log()
        values.append(dist.base.log_prob(preimage) - jac)
    expected = torch.logsumexp(torch.stack(values), 0)
    actual = dist.log_prob(exp_origin(v, k))
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    assert actual - values[80] > .1
    unresolved = ProjectedWrappedNormal(torch.zeros(2, dtype=dtype), torch.full((2,), 30., dtype=dtype),
                                         k, max_windings=1)
    try:
        unresolved.log_prob(exp_origin(v, k))
    except FloatingPointError:
        pass
    else:
        raise AssertionError("unresolved spherical density was accepted")


def check_product_and_flat_limit():
    torch.manual_seed(10)
    flat = PaperFlatVAE(latent_dim=6, encoder_width=8, decoder_width=8).double()
    model = MixedCurvatureVAE(latent_dim=6, encoder_width=8, decoder_width=8, mode="universal").double()
    model.encoder.load_state_dict(flat.encoder.state_dict())
    model.decoder.load_state_dict(flat.decoder.state_dict())
    x = torch.randn(8, 3, dtype=torch.float64)
    for method in ("posterior", "prior"):
        a = getattr(flat, method)(x) if method == "posterior" else flat.prior()
        b = getattr(model, method)(x) if method == "posterior" else model.prior()
        torch.manual_seed(11)
        z = a.rsample((3,))
        torch.manual_seed(11)
        actual = b.rsample((3,))
        torch.testing.assert_close(z, actual)
        torch.testing.assert_close(a.log_prob(z), b.log_prob(z))
        torch.testing.assert_close(flat.observation(z).mean, model.observation(z).mean)
    torch.manual_seed(12)
    expected = elbo_loss(flat, x)
    torch.manual_seed(12)
    torch.testing.assert_close(elbo_loss(model, x), expected)
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in flat.parameters()) + 3

    model = MixedCurvatureVAE(latent_dim=7, factor_dims=[2, 3, 2], encoder_width=8, decoder_width=8).double()
    q, p = model.posterior(x), model.prior()
    z = q.rsample((3, 2))
    assert z.shape == (3, 2, 8, 7)
    torch.testing.assert_close(q.log_prob(z), q.factor_log_prob(z).sum(-1))
    assert torch.isfinite(p.log_prob(z)).all()
    torch.testing.assert_close(model.factor_distances(z, z), torch.zeros_like(q.factor_log_prob(z)), atol=1e-8, rtol=0)
    # Full product Jacobian, including independent volume factors.
    vector = torch.linspace(.1, .3, 7, dtype=torch.float64)
    dist = ProductWrappedNormal(q.location[0], torch.full((7,), .2, dtype=torch.float64),
                                model.curvatures, model.factor_dims)
    point = dist.wrap(vector)
    jacobian = torch.autograd.functional.jacobian(dist.wrap, vector)
    volume = torch.linalg.slogdet(jacobian).logabsdet
    for part, k, d in zip(point.split(model.factor_dims), model.curvatures, model.factor_dims):
        volume -= d * torch.log1p(k * part.square().sum() / 4)
    torch.testing.assert_close(dist.log_prob(point), dist.base.log_prob(vector) - volume)


def check_gradients():
    for mode in ("mixed", "universal"):
        for initial in ((0., 0.), (-1e-6, 1e-6), (-.5, .5)) if mode == "universal" else ((-1., 1.),):
            torch.manual_seed(13)
            model = MixedCurvatureVAE(latent_dim=4, encoder_width=8, decoder_width=8, mode=mode).double()
            if mode == "universal":
                with torch.no_grad():
                    for factor, k in zip(model.factors, initial):
                        factor.signed_curvature.fill_(k)
            x = torch.randn(5, 3, dtype=torch.float64)
            def objective():
                torch.manual_seed(14)
                return elbo_loss(model, x)
            objective().backward()
            for parameter in model.geometry_parameters():
                original = parameter.detach().clone()
                gradient = parameter.grad.clone()
                assert torch.isfinite(gradient) and gradient.abs() > 1e-8
                with torch.no_grad():
                    parameter.copy_(original + 1e-7)
                    plus = objective()
                    parameter.copy_(original - 1e-7)
                    minus = objective()
                    parameter.copy_(original)
                torch.testing.assert_close(gradient, (plus - minus) / 2e-7, rtol=2e-4, atol=2e-5)

    # Real ELBO updates can change each universal component's sign in both directions.
    directions = [set(), set()]
    for seed in range(20):
        torch.manual_seed(seed)
        model = MixedCurvatureVAE(latent_dim=4, encoder_width=8, decoder_width=8, mode="universal",
                                  curvature_warmup_epochs=0, curvature_stabilization_epochs=0).double()
        model.set_training_epoch(1)
        parameters = model.geometry_parameters()
        x = torch.randn(5, 3, dtype=torch.float64)
        with torch.no_grad():
            for parameter in parameters:
                parameter.zero_()
        torch.manual_seed(30)
        elbo_loss(model, x).backward()
        starts = [parameter.grad.sign() * 1e-7 for parameter in parameters]
        with torch.no_grad():
            for parameter, start in zip(parameters, starts):
                parameter.copy_(start)
        model.zero_grad()
        torch.manual_seed(30)
        elbo_loss(model, x).backward()
        model.prepare_geometry_step()
        torch.optim.SGD(parameters, lr=.001).step()
        for i, (parameter, start) in enumerate(zip(parameters, starts)):
            assert start * parameter < 0
            directions[i].add(int(start.sign()))
        if all(d == {-1, 1} for d in directions):
            break
    assert all(d == {-1, 1} for d in directions)


def check_schedule_and_resume():
    points = np.random.default_rng(15).normal(size=(12, 3)).astype(np.float32)
    def make_model():
        torch.manual_seed(16)
        return MixedCurvatureVAE(latent_dim=6, encoder_width=8, decoder_width=8, mode="universal",
                                  curvature_warmup_epochs=2, curvature_stabilization_epochs=1)
    with TemporaryDirectory() as directory:
        path = Path(directory)
        model, full = make_model(), make_model()
        train(model, points, points[:4], path / "resume", epochs=2, batch_size=4)
        history = train(model, points, points[:4], path / "resume", epochs=5, batch_size=4)
        expected = train(full, points, points[:4], path / "full", epochs=5, batch_size=4)
        assert history == expected
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, full.state_dict()[key], atol=0, rtol=0)
        assert history[1]["curvatures"] == [0., 0., 0.]
        np.testing.assert_allclose(history[2]["curvatures"], [-.001, .001, 0.])
        assert history[2]["curvature_stage"] == "stabilization"
        assert history[3]["curvature_stage"] == "learned"
        assert history[3]["curvatures"][:2] != history[2]["curvatures"][:2]
        assert all(row["curvatures"][2] == 0 for row in history)
        best = torch.load(path / "resume" / "best.pt", weights_only=True)
        assert best["epoch"] >= 4
        assert "geometry_optimizer" in best
        # Resume after perturbation must not perturb or reset the learned geometry again.
        train(model, points, points[:4], path / "resume", epochs=6, batch_size=4)
        train(full, points, points[:4], path / "full", epochs=6, batch_size=4)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, full.state_dict()[key], atol=0, rtol=0)


def check_validation():
    assert partition_dimensions(4) == (2, 2) and partition_dimensions(7) == (2, 2, 3)
    for settings in ({"latent_dim": 1}, {"factor_dims": [1, 5]}, {"factor_types": ["h", "s"]},
                     {"curvatures": [1., 1., 0.]}, {"curvatures": [-1., float("nan"), 0.]},
                     {"mode": "universal", "curvatures": [-1., 1., 0.]},
                     {"mode": "universal", "learn_curvature": False}):
        try:
            MixedCurvatureVAE(**settings)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid settings accepted: {settings}")
    fixed = MixedCurvatureVAE(learn_curvature=False)
    assert not fixed.geometry_parameters()
    try:
        MixedCurvatureVAE(factor_dims=[3, 3], factor_types=["h", "s"]).load_state_dict(fixed.state_dict())
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched checkpoint signature accepted")


def check_likelihood_integration():
    # The new models must work with the independent 2D prior integration path.
    x = np.array([[.1, -.2, .3], [0., .1, -.1]], dtype=np.float64)
    for kind in ("e", "h", "s"):
        model = MixedCurvatureVAE(latent_dim=2, factor_types=[kind], encoder_width=4,
                                  decoder_width=4, observation_std=.3, learn_curvature=False).double()
        with torch.no_grad():
            for parameter in model.decoder.parameters():
                parameter.zero_()
        metrics, arrays = evaluate(model, x, importance_samples=16, prior_fraction=1., quadrature_grid_size=40)
        expected = torch.distributions.Independent(torch.distributions.Normal(
            torch.zeros_like(torch.from_numpy(x)), model.observation_std), 1).log_prob(torch.from_numpy(x))
        torch.testing.assert_close(torch.from_numpy(arrays["log_likelihood"]), expected)
        torch.testing.assert_close(torch.from_numpy(arrays["iwll"]), expected)
        assert metrics["likelihood_checked"] is True


def check_experiments():
    from run import main
    with TemporaryDirectory() as directory:
        settings = dict(OUTPUT_DIR=directory, MODEL_TYPES=["paper_flat", "mixed", "paper_universal"],
                        LATENT_DIMS=[6], DECODER_WIDTHS=[8], SEEDS=[0, 1], DATASETS=["sphere"],
                        DATA={**config.DATA, "n_samples": 40}, MODEL={**config.MODEL, "encoder_width": 8},
                        MIXED_MODEL={"learn_curvature": True},
                        PAPER_UNIVERSAL_MODEL={"curvature_warmup_epochs": 1, "curvature_stabilization_epochs": 1},
                        TRAINING={**config.TRAINING, "epochs": 4, "batch_size": 16},
                        EVALUATION={**config.EVALUATION, "importance_samples": 16, "sample_chunk_size": 7})
        with patch.multiple(config, **settings), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            main()
            config.TRAINING = {**config.TRAINING, "epochs": 5}
            main()
        path = Path(directory)
        summary = json.loads((path / "summary.json").read_text())
        assert len(summary) == 3 and all(r["n_seeds"] == 2 for r in summary)
        results = [json.loads(p.read_text()) for p in path.glob("*/results.json")]
        assert len(results) == 6 and all(r["latent_dim"] == 6 for r in results)
        assert all(r["likelihood_checked"] is False for r in results)
        for file in path.glob("*/results.json"):
            result = json.loads(file.read_text())
            if result["model_type"] == "MixedCurvatureVAE":
                assert result["curvature"] is None and len(result["curvatures"]) == 3
                assert result["curvature_stage"] == "learned"
                assert len(result["factor_kl"]) == 3
                assert (file.parent / "curvatures.png").exists()
                assert (file.parent / "factor_slices.png").exists()
                with np.load(file.parent / "test_metrics.npz") as archive:
                    np.testing.assert_allclose(archive["factor_kl"].sum(-1), archive["kl"])
                    assert archive["latent_location"].shape[-1] == 6
                checkpoint = torch.load(file.parent / "best.pt", weights_only=True)
                model = MixedCurvatureVAE(encoder_width=8, decoder_width=8,
                    mode="universal" if result["model_variant"] == "paper_universal" else "mixed",
                    curvature_warmup_epochs=1, curvature_stabilization_epochs=1)
                model.load_state_dict(checkpoint["model"])
                np.testing.assert_allclose(model.curvatures.detach().numpy(), result["curvatures"])
        assert (path / "iwll.png").exists() and (path / "results.csv").exists()


def check_paper():
    check_maps_and_jacobians()
    check_normalization_and_windings()
    check_product_and_flat_limit()
    check_gradients()
    check_schedule_and_resume()
    check_validation()
    check_likelihood_integration()
    check_experiments()


if __name__ == "__main__":
    check_paper()
    print("Paper product geometry, normalized wrapped densities, gradients, schedule, resume and experiments passed.")
