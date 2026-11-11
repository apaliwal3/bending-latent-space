# Bending latent space

Compare flat and curved VAE latent spaces on noisy sphere, Swiss roll and saddle data.

Edit **`config.py`**, then run:

```bash
uv sync
uv run python run.py
```

The lists at the top of `config.py` control model types, datasets, noise, latent
dimensions, decoder widths and seeds. Every combination runs. The default compares
one flat VAE and one curved VAE on the same sphere data and splits.

```python
MODEL_TYPES = ["flat", "curved"]  # Use either entry alone to run one model.
CURVED_MODEL = {"curvature": -1.0, "learn_curvature": True}
```

`VAE` uses a Euclidean diagonal-Gaussian posterior and standard-normal prior.
`CurvedVAE` uses a hyperbolic latent space with fixed or learned negative sectional curvature
`K`. For `latent_dim=d`, its latent samples contain `d+1` Lorentz coordinates on
the upper hyperboloid, with Minkowski norm squared `1/K`. Encoder widths,
decoder widths match the flat baseline. Learning curvature adds one scalar parameter.

The curved prior and posterior use the
[wrapped normal of Nagano et al. (2019)](https://proceedings.mlr.press/v97/nagano19a.html).
Sampling draws a diagonal Gaussian at the origin tangent space, parallel transports
it to the encoded location, and applies the exponential map. `log_prob` inverts
these maps and subtracts the exponential-map log volume:

```text
(d - 1) * log(sinh(sqrt(-K) * r) / (sqrt(-K) * r))
```

Here `r` is the tangent vector's geodesic length. Both latent densities use
Riemannian volume, so sampled KL, ELBO and IWLL use the same reference measure.
The decoder maps samples back to origin tangent coordinates before its MLP and
retains the Gaussian observation likelihood in 3D. This gives a curved latent
distribution while keeping the decoder architecture comparable.

With `learn_curvature=True`, `curvature` is the initial value. The model learns
one global scalar via `K = -(softplus(raw_curvature) + 1e-6)`. Adam automatically
includes this parameter through `model.parameters()`; gradients flow through
sampling, geometry maps, and both latent densities. Set `learn_curvature=False`
for fixed curvature (also the constructor default). Learned runs require an
initial curvature below `-1e-6`. This implementation covers negative
curvature; it does not implement spherical or spatially varying latent curvature.
The sphere dataset can still be used, but its positive surface curvature is not
the model's latent curvature. In one intrinsic dimension there is no sectional
curvature comparison. The curved encoder clamps tangent log variance to `[-10, 2]`
for numerical stability. Geometry uses float64 on CPU/CUDA and float32 on MPS;
large hyperbolic distances can lose precision, especially on MPS.

You can also construct either model directly:

```python
from model import CurvedVAE, VAE

flat = VAE(latent_dim=2)
curved = CurvedVAE(latent_dim=2, curvature=-1.0, learn_curvature=True)
# Both expose posterior(x), prior(), observation(z), and latent_from_tangent(v).
```

- `data.py`: generate 3D points, fixed splits and ground-truth surface curvature.
- `model.py`: flat VAE, hyperbolic maps, wrapped normal and curved VAE.
- `train.py`: shared ELBO training and checkpoint saving/loading.
- `evaluate.py`: held-out IWLL and reconstruction metrics.
- `plot.py`: loss curves, point clouds, seed comparisons and decoder surfaces.
- `run.py`: run experiments and save results/plots in `outputs/experiment/`.

Rerunning resumes saved training; increase `TRAINING["epochs"]` to train longer.
For other changes to fixed settings, choose a new `OUTPUT_DIR`. Existing flat
experiment directories remain compatible; curved directories include their model
type and initial curvature; learned runs have a `learned_` prefix, keeping them
separate from existing fixed runs. Curvature is saved each epoch in `history.json`.
`results.json` reports the curvature from the best validation checkpoint.
`summary.json` groups seeds by model, initial curvature, learning mode, dataset,
noise, latent dimension and decoder width, and reports learned curvature values,
means and standard deviations. IWLL plots distinguish fixed and learned models.
Learning latent curvature does not establish recovery of data-surface curvature.
Decoder surfaces use an origin-tangent grid; optional volume coloring measures
the full decoder map in those grid coordinates, not intrinsic latent volume.

Run the correctness checks with:

```bash
uv run python check.py
```

These check data reproducibility, exact linear-Gaussian IWLL, manifold constraints,
the wrapping Jacobian/density, gradients at the origin, the flat-curvature limit,
curved training/evaluation, curvature gradients, updates and checkpoint restoration.
