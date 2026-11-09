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



For this project, I'd outsource most of the engineering and repetitive work to me, while keeping the scientific decisions and interpretation between the two of you.

| Task                                                   | Outsource to me?     |
| ------------------------------------------------------ | -------------------- |
| Synthetic data generators (sphere, Swiss roll, saddle) | &#x20;Fully          |
| Data splits, noise injection, reproducibility          | &#x20;Fully          |
| Experiment harness, checkpointing, logging             | &#x20;Fully          |
| Training loops and baseline VAE boilerplate            | &#x20;Mostly         |
| Hyperparameter sweep configuration                     | &#x20;Fully          |
| Evaluation metrics (IWLL, KL, etc.)                    | &#x20;Implementation |
| Plotting and visualization code                        | &#x20;Fully          |
| Curved VAE and wrapped normal implementation           | &#x20;Collaborative  |
| Mathematical derivations and numerical debugging       | &#x20;Collaborative  |
| Running large GPU experiments                          | &#x20;You two        |
| Experimental design and scientific hypotheses          | &#x20;You two        |
| Interpreting results and drawing conclusions           | &#x20;You two        |

I'd particularly recommend outsourcing three things:

1. Experiment infrastructure: It's necessary but rarely intellectually rewarding.
2. Plotting and reporting: I can generate the visualization code, summary tables, and statistical comparisons.
3. Routine implementation: Data generators, standard training loops, checkpointing, and tests.

That leaves both of you focusing on the interesting questions: implementing and understanding curved latent geometry, understanding why the models behave differently, and designing experiments that distinguish genuine curvature benefits from decoder flexibility.

One caveat: I can write and test code, but you should understand and review the mathematical assumptions, particularly the probability densities and evaluation metric. A subtle error there could invalidate the entire experiment.

My recommendation is to divide the scientific work between you two, rather than dividing the boring engineering work.