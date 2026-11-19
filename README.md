# Bending latent space

Compare flat, constant-curvature and mixed-curvature product VAE latent spaces
on noisy sphere, Swiss roll and saddle data.

Edit **`config.py`**, then run:

```bash
uv sync
uv run python run.py
```

Every combination of model types, datasets, noise, total latent dimensions,
decoder widths and seeds runs. Defaults now compare the diagonal wrapped-normal
product model from [Mixed-curvature Variational Autoencoders (ICLR 2020)](https://arxiv.org/abs/1911.08411)
against a matched flat control:

```python
MODEL_TYPES = ["paper_flat", "mixed", "paper_universal"]
LATENT_DIMS = [6]
```

- `paper_flat`: 6D Euclidean VAE with the same network and posterior scale
  parameterization as the paper models.
- `mixed`: 2D hyperbolic x 2D spherical x 2D Euclidean factors. Each curved
  factor has its own radius, optionally learned with SGD. Set
  `MIXED_MODEL["learn_curvature"] = False` for a fixed-curvature control.
- `paper_universal`: three initially flat 2D factors. Train flat for 50 epochs,
  perturb the factors to negative/positive/zero curvature, stabilize for five
  epochs, then learn the two non-Euclidean signed curvatures. Either can cross
  zero; the assigned Euclidean factor stays flat.

`latent_dim` always denotes the **total intrinsic dimension**, so baselines and
products are compared at the same dimension. `factor_dims=None` partitions it
into 2D factors, using one 3D factor for an odd total. For a two-factor 4D model:

```python
LATENT_DIMS = [4]
MIXED_MODEL = {
    "factor_dims": [2, 2],
    "factor_types": ["h", "s"],
    "curvatures": [-1., 1.],
    "learn_curvature": True,
}
```

The new implementation uses Gaussian tangent samples, parallel transport and
the geodesic exponential, rather than the original custom radial distribution.
For spherical factors it includes winding contributions with a checked tail
bound. This corrects the reference implementation's single inverse branch.
The networks and likelihood are adapted to this project's 3D data; this is not
a reproduction of the paper's image benchmark results. See [PAPER_MODEL.md](PAPER_MODEL.md)
for equations, paper correspondence, analysis outputs and explicit differences.

Results are saved under `outputs/paper_mixed/`. Each product run saves
`curvatures.png`, `factor_slices.png`, per-factor KL contributions, posterior
locations/scales and the usual likelihood/reconstruction metrics. Curvatures
are reported as vectors, not as a single product curvature. Use multiple
`SEEDS` for comparisons; the default one-seed run is a demonstration. Likelihood
estimates above total dimension two remain marked `likelihood_checked=False`.

The original single-factor signed model remains available with:

```python
MODEL_TYPES = ["flat", "universal"]
LATENT_DIMS = [2]
UNIVERSAL_MODEL = {"curvature": 0.0, "learn_curvature": True}
```

`UniversalVAE` learns `K` directly as an unconstrained scalar parameter. Negative
values mean hyperbolic geometry, zero means Euclidean geometry, and positive
values mean spherical geometry. Gradients exist at zero and can move curvature
in either direction. Set `learn_curvature=False` to fix any finite signed value.
Exact zero is supported, but optimization does not automatically round a small
learned value to zero. This is a single constant-curvature space, not a model of
spatially varying curvature or a product of different geometries.

```python
from model import UniversalVAE

model = UniversalVAE(latent_dim=2, curvature=0.0, learn_curvature=True)
# posterior(x), prior(), observation(z), latent_from_tangent(v)
```

The signed model uses `d` stereographic coordinates for intrinsic dimension `d`.
The metric is `g(z) = (1 + K * ||z||^2 / 4)^(-2) I`. For negative curvature the
chart is a ball of radius `2/sqrt(-K)`; at zero it is Euclidean space; for positive
curvature it represents a sphere of radius `1/sqrt(K)` with one pole omitted.
Stereographic geometry avoids divergent Lorentz coordinates at zero; see
[Mixed-curvature Variational Autoencoders](https://arxiv.org/abs/1911.08411)
for the geometric framework. The distribution below is our explicit bijective
Gaussian pushforward, not that paper's exponential-map wrapped distribution.

For a Gaussian vector `v` with `r = ||v||`, define the radial chart transform:

```text
K < 0: F_K(v) = 2v / (1 + sqrt(1 - K*r^2))
K >= 0: F_K(v) = (1 + K*r^2/4) * v
```

Each branch is one-to-one onto the entire chart domain. Their values and first
curvature derivatives agree at zero (the family is C1 there). The posterior
translates `F_K(v)` to the encoded center with a Möbius isometry. The prior uses
the origin center and unit Gaussian scale. `log_prob` includes both the radial
Jacobian and Riemannian volume correction. The spherical prior is not uniform.
This construction avoids ambiguous spherical exponential-map inverses and
omitted winding contributions, while keeping normalized densities and full
manifold support up to sets of zero volume. At zero it recovers a Gaussian VAE.

The decoder uses the geodesic logarithm at the origin followed by the same MLP
and Gaussian 3D likelihood as the baseline. Positive-curvature charts and the
origin logarithm have pole/antipode singularities; the omitted points have zero
probability, but nearby points can still cause numerical conditioning problems.
Geometry uses float64 on CPU/CUDA and float32 on MPS. Large coordinate values or
points near the hyperbolic boundary also lose precision. Encoder log variance
is clamped to `[-10, 2]`. GPU execution has not been verified.

The original models remain available:

- `"flat"`: `VAE`, Euclidean diagonal Gaussian prior/posterior.
- `"curved"`: `CurvedVAE`, negative-only Lorentz wrapped normals with
  `CURVED_MODEL` settings; learnable curvature stays negative. This follows
  [Nagano et al. (2019)](https://proceedings.mlr.press/v97/nagano19a.html).
- `"universal"`: `UniversalVAE`, signed curvature with `UNIVERSAL_MODEL` settings.

Switching from `curved` to `universal` changes the distribution family and latent
representation. Their checkpoints are separate and cannot be interchanged.
Learning adds one scalar parameter to the otherwise matched encoder/decoder
architectures. Learned latent curvature is not automatically a recovery of the
observed data surface's Gaussian curvature. In intrinsic dimension one there is
no sectional curvature to compare.

- `data.py`: reproducible 3D data, splits and ground-truth surface curvature.
- `geometry.py`: signed stereographic geometry and normalized distributions.
- `paper_geometry.py`: exponential-map wrapped normals and product distributions.
- `mixed.py`: matched paper baseline, mixed products and universal schedule.
- `model.py`: model definitions and the original Lorentz geometry.
- `train.py`: shared ELBO training, checkpoints and per-epoch curvature history.
- `evaluate.py`: held-out likelihood and reconstruction metrics.
- `plot.py`: loss curves, point clouds, seed comparisons and decoder surfaces.
- `run.py`: experiments and results/plots under the configured `OUTPUT_DIR`.

Rerunning resumes saved training; increase `TRAINING["epochs"]` to train longer.
For other changes to fixed settings, choose a new `OUTPUT_DIR`. Experiment paths
include model type and initial curvature; learned runs have a `learned_` prefix.
`history.json` records curvature each epoch; `results.json` reports it from the
best validation checkpoint. `summary.json` groups seeds by model, initial
curvature, learning mode, dataset, noise, latent dimension and decoder width,
and reports curvature values, means and standard deviations. Likelihood plots keep
model families separate. Decoder surfaces map an origin-coordinate grid through
the model's `latent_from_tangent`; volume coloring measures the full decoder map
in those grid coordinates, not intrinsic latent volume.

Likelihood comparisons use `log_likelihood`. For 2D models this integrates over
the prior on CPU, checks a doubled grid, and bounds omitted prior tails. An
unresolved integral raises an error. Other dimensions use an importance estimate
mixing posterior and prior samples; these have `likelihood_checked=False` and
are marked with crosses in plots. `iwll` and effective sample sizes are also
saved. Low effective sample size or disagreement with integration produces a
warning. More samples or a high effective sample size alone do not establish
accuracy; higher-dimensional rankings still need independent validation.

Run correctness checks with:

```bash
uv run python check.py
# Only the paper extension:
uv run python check_paper.py
```

Checks cover data reproducibility, exact linear-Gaussian IWLL, original Lorentz
geometry, signed-map inverses and Jacobians, independent numerical density
normalization, the exact flat case, finite-difference curvature gradients at and
around zero, optimizer sign crossing in both directions and checkpoint restoration.
The paper extension additionally checks ambient-space agreement, product-map
Jacobians, independent wrapped-density normalization including spherical winding,
curvature gradients/sign crossing, the matched flat limit, stage transitions,
exact optimizer/RNG resume and two-seed end-to-end experiment reporting.
