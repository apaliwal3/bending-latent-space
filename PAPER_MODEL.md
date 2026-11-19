# Mixed-curvature model

This project implements the diagonal wrapped-normal product model of Skopek,
Ganea and Becigneul, *Mixed-curvature Variational Autoencoders*, ICLR 2020:
[paper, version 2](https://arxiv.org/pdf/1911.08411v2),
[authors' implementation](https://github.com/oskopek/mvae).

It is an independent implementation of the geometric model, adapted to the
existing 3D surface experiments. The original `UniversalVAE` remains a separate
model with a different, bijective radial distribution.

## Model and paper correspondence

| Paper | Implementation |
| --- | --- |
| Section 2.3: product of constant-curvature factors | `MixedCurvatureVAE`, arbitrary factor count and dimensions |
| Sections 2.2/2.4.1: Gaussian, transport, exponential | `ProjectedWrappedNormal` in `paper_geometry.py` |
| Section 3: factorized prior/posterior, concatenated latent | `ProductWrappedNormal`, direct chart-coordinate decoder |
| Section 3.1: learn radii with SGD | `mode="mixed"`, radius per non-Euclidean factor |
| Section 3.1: universal curvature procedure | `mode="universal"`, staged flat/perturb/stabilize/learn |
| Reference feedforward network | One hidden ReLU layer in encoder and decoder |
| Reference posterior scale | Diagonal `softplus(raw_scale) + 1e-5` |

`factor_types` uses `h` for projected hyperbolic, `s` for projected spherical,
and `e` for Euclidean geometry. These are the paper's Poincare/projected sphere
realizations; they represent the same constant-curvature manifolds as the
Lorentz/hypersphere realizations. Extrinsic Lorentz/hypersphere decoder variants,
vMF distributions and Riemannian normals are not added by this extension.

For dimensions `d_1, ..., d_m`,

```text
M = M_1(K_1) x ... x M_m(K_m),  D = sum_i d_i
p(z) = product_i p_i(z_i)
q(z | x) = product_i q_i(z_i | x)
KL(q || p) = sum_i KL(q_i || p_i)
```

Each curved factor has dimension at least two. Euclidean factors may be 1D.
An automatic partition has 2D factors and one 3D factor when D is odd. The
dimensions/signature are chosen in configuration, not learned.

The encoder produces one origin-tangent mean and diagonal scale per factor.
Its mean is exponentiated at the origin. To sample a posterior, draw a centered
tangent Gaussian, transport it to the mean and exponentiate there. Priors use
the origin and unit Gaussian scale. The decoder consumes the concatenation of
projected points. `latent_from_tangent` maps standard normal vectors to prior
samples, including all spherical windings.

## Coordinate convention and density

We use coordinates `z = 2 * y`, where `y` denotes the paper's projected chart.
The equivalent metric is

```text
g_z = (1 + K ||z||^2 / 4)^(-2) I.
```

This keeps sectional curvature K and makes the metric at the origin I. Tangent
Gaussian coordinates are orthonormal, so the zero-curvature model matches the
flat Gaussian control. For `r = ||v||`, the origin exponential is

```text
K < 0: exp_0(v) = tanh(sqrt(-K)*r/2) / (sqrt(-K)*r/2) * v
K = 0: exp_0(v) = v
K > 0: exp_0(v) = tan(sqrt(K)*r/2) / (sqrt(K)*r/2) * v.
```

Transport from the origin followed by exponentiation at a mean mu can be
written `mu (+)_K exp_0(v)`, with Mobius addition. Its Riemannian Jacobian is
`[sin(sqrt(K)*r)/(sqrt(K)*r)]^(d-1)` for positive curvature, using sinh for
negative curvature and the continuous limit for zero curvature.

On a sphere the exponential is many-to-one. If the shortest inverse has
radius r and direction u, its preimages include
`(r + 2*pi*n/sqrt(K)) * u` for every integer n. The reference code uses one
branch; that omits density mass when tangent samples wrap around the sphere.
Our density sums these branches. Each omitted half-series is bounded using the
first omitted Gaussian-times-polynomial term and a decreasing geometric ratio.
The bound must be at most `winding_tolerance` times the retained density.
An unresolved sum raises `FloatingPointError`. Configure `winding_tolerance`
(default `1e-10`) and `max_windings` (default 64) in model settings if needed.

This is a deliberate correction for normalized likelihood analysis, rather
than a bit-for-bit port of the authors' code. Densities can be singular at
spherical conjugate points, which have probability zero. Chart poles and
hyperbolic boundaries can still cause numerical conditioning problems.
Geometry runs in float64 on CPU/CUDA and float32 on MPS; GPU runs have not
been validated. The winding bound controls truncation, not floating-point error.

## Training and checkpoints

`mixed` holds the signature fixed. Set initial curvatures by factor. With
`learn_curvature=True`, it optimizes positive radii by SGD:
`K_i = sign_i / R_i^2`. Euclidean factors stay at zero. Radii are projected to
at least `1e-4` after updates to avoid invalid geometry. Geometry gradient norm
is clipped to one; network parameters use Adam. Set
`TRAINING["curvature_learning_rate"]` independently of the network learning rate.

`paper_universal` begins with zero curvatures. The default 100-epoch schedule is:

1. Epochs 1–50: flat factors; train encoder/decoder.
2. Epoch 51: perturb components to `[-epsilon, +epsilon, 0]`, repeated by factor.
3. Epochs 51–55: keep curvature fixed while stabilizing the networks.
4. Epochs 56–100: optimize the signed curvatures of the perturbed components.

The grouping is approximately equal in size; for two factors it becomes h+s.
The assigned Euclidean group remains fixed. Each other component can cross
zero independently. Warmup, stabilization and epsilon are explicit model
settings. Unlike a schedule derived from the requested total epochs, their
boundaries stay stable when a checkpoint is resumed for longer training.
The runner requires enough epochs to reach the last stage.

Both optimizer states, the geometry stage and RNG state are checkpointed.
Resuming does not repeat perturbation. Best-validation selection resets at
phase transitions so a final universal result does not silently restore the
earlier flat warmup. The factor signature is saved and checked on load.

Network sizes use this project's configurable widths instead of the paper's
400-neuron image experiments. Observations use the existing fixed-scale 3D
Gaussian. Training uses the shared ELBO and validation-based checkpoint
selection, rather than the paper's benchmark early stopping on training ELBO.
Optional `beta_warmup_epochs` linearly anneals the training KL weight; default
zero uses the ordinary ELBO throughout. Validation always uses the full ELBO.

## Analysis outputs

The default `paper_flat` is the matched control: same total dimension, network
architecture, diagonal scale parameterization and observation likelihood.
Fixed products are available by setting `MIXED_MODEL["learn_curvature"]=False`.
The older `flat` control belongs to the older two-layer Tanh architecture;
use `paper_flat` for the new comparison. All configurations share generated
data and splits; use several seeds for uncertainty estimates.

Each product run saves:

- `config.json`: resolved dimensions/types, initial curvatures and schedule.
- `history.json`, `curvatures.png`: each factor's curvature and training stage.
- `results.json`: best-checkpoint curvature vector, factor KL means, parameter
  counts and reconstruction/likelihood metrics. Scalar curvature is null.
- `test_metrics.npz`: point indices, factor KL samples, posterior chart centers
  (`latent_location`) and tangent scales (`latent_scale`). Centers are not
  manifold expectations, and scales are not chart-coordinate standard deviations.
- `factor_slices.png`: vary the first two tangent directions of each factor,
  holding all other coordinates at zero. For a factor above 2D this is a slice.
- `summary.json`, `results.csv`, `iwll.png`: comparisons grouped by model variant,
  factor signature, initial curvature vector, total dimension, width and dataset.

The API `model.factor_distances(z, w)` returns geodesic distances by factor;
the product distance is their squared sum's square root.

Likelihoods integrate the prior directly for total dimension two. Above two,
they remain finite-sample importance estimates and are marked
`likelihood_checked=False`. Density normalization tests do not certify those
estimates. Effective sample sizes are reported, and low values warn. The
existing evaluator mixes posterior and prior proposals by default; set
`EVALUATION["prior_fraction"]=0` for the paper's posterior-only estimator and
increase `importance_samples` (the paper uses 500 for most experiments).

Product curvature depends on tangent plane: planes wholly within a factor
have K_i, and mixed planes have zero curvature. This is not arbitrary spatially
varying K(z). Factor labels can permute across seeds. Neither learned latent
curvature nor decoder slices establish recovery of the observed surface's
Gaussian curvature. Sphere/Swiss-roll/saddle data are demonstrations; the
paper's BDP/image benchmark results are not reproduced here.

## Validation

```bash
uv run python check.py
uv run python check_paper.py  # Only this extension.
```

Checks compare projected operations against independent ambient Lorentz and
sphere implementations, differentiate full product Jacobians, integrate wrapped
densities against manifold volume (including multiple spherical windings), and
compare adaptive sums with long explicit sums. They also check both curvature
gradients, independent sign crossing, the exact flat limit in float64,
signature rejection, staged training, exact resume across phase transitions,
and two-seed runs through evaluation, saved outputs, plots and resume.
