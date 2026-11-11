# Bending latent space

Compare flat and curved VAE latent spaces on noisy sphere, Swiss roll and saddle data.

Edit **`config.py`**, then run:

```bash
uv sync
uv run python run.py
```

Every combination of model types, datasets, noise, latent dimensions, decoder
widths and seeds runs. The default compares a flat VAE with a model that learns
one global signed sectional curvature, starting at exactly zero:

```python
MODEL_TYPES = ["flat", "universal"]
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
- `model.py`: model definitions and the original Lorentz geometry.
- `train.py`: shared ELBO training, checkpoints and per-epoch curvature history.
- `evaluate.py`: held-out IWLL and reconstruction metrics.
- `plot.py`: loss curves, point clouds, seed comparisons and decoder surfaces.
- `run.py`: experiments and results/plots in `outputs/experiment/`.

Rerunning resumes saved training; increase `TRAINING["epochs"]` to train longer.
For other changes to fixed settings, choose a new `OUTPUT_DIR`. Experiment paths
include model type and initial curvature; learned runs have a `learned_` prefix.
`history.json` records curvature each epoch; `results.json` reports it from the
best validation checkpoint. `summary.json` groups seeds by model, initial
curvature, learning mode, dataset, noise, latent dimension and decoder width,
and reports curvature values, means and standard deviations. IWLL plots keep
model families separate. Decoder surfaces map an origin-coordinate grid through
the model's `latent_from_tangent`; volume coloring measures the full decoder map
in those grid coordinates, not intrinsic latent volume.

Run correctness checks with:

```bash
uv run python check.py
```

Checks cover data reproducibility, exact linear-Gaussian IWLL, original Lorentz
geometry, signed-map inverses and Jacobians, independent numerical density
normalization, the exact flat case, finite-difference curvature gradients at and
around zero, optimizer sign crossing in both directions and checkpoint restoration.
