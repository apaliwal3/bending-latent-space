# Bending latent space

Compare flat and curved VAE latent spaces on noisy sphere, Swiss roll and saddle data.

Edit **`config.py`**, then run:

```bash
uv sync
uv run python run.py
```

The lists at the top of `config.py` control datasets, noise, latent dimensions,
decoder widths and seeds. Every combination runs; the default is one experiment.

- `data.py`: generate 3D points, fixed splits and ground-truth curvature.
- `model.py`: the flat Gaussian VAE; edit the encoder and decoder here.
- `train.py`: ELBO training and checkpoint saving/loading.
- `evaluate.py`: held-out IWLL and reconstruction metrics.
- `plot.py`: loss curves, point clouds, seed comparisons and decoder surfaces.
- `run.py`: run experiments, evaluate and save results/plots in `outputs/experiment/`.

Rerunning resumes saved training; increase `TRAINING["epochs"]` to train longer.
For other changes to fixed settings, choose a new `OUTPUT_DIR`.
`summary.json` keeps individual seed scores, means and standard deviations.
Run `uv run python check.py` for two small correctness checks.

Add curved model through `posterior(x)`, `prior()` and `observation(z)`;
replace the `VAE` construction in `run.py`. Log probabilities sum event dimensions.
