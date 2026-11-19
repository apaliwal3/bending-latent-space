"""Edit these settings, then run: uv run python run.py"""

OUTPUT_DIR = "outputs/paper_mixed"

# Every combination runs. Add values here to compare experiments.
DATASETS = ["sphere"]
MODEL_TYPES = ["paper_flat", "mixed", "paper_universal"]
NOISE_LEVELS = [0.05]
LATENT_DIMS = [6]  # Total dimension: 2D hyperbolic + 2D spherical + 2D flat.
DECODER_WIDTHS = [32]
SEEDS = [0]

DATA = {
    "n_samples": 6000,
    "seed": 42,  # Same data for every model seed.
    "split_seed": 43,
    "train_fraction": 0.7,
    "val_fraction": 0.15,
}
SURFACES = {
    "sphere": {"radius": 1.0},
    "swiss_roll": {"t_min": 4.71, "t_max": 14.14, "height": 2.0, "radius_scale": 0.1},
    "saddle": {"extent": 1.0, "a": 0.5, "b": 0.5},
}
MODEL = {
    "encoder_width": 64,
    "observation_std": 0.1,  # Likelihood scale, separate from dataset noise.
}
CURVED_MODEL = {"curvature": -1.0, "learn_curvature": True}  # Initial curvature.
UNIVERSAL_MODEL = {"curvature": 0.0, "learn_curvature": True}  # Can cross zero.
MIXED_MODEL = {
    "factor_dims": None,  # Paper partition: 2D factors, one 3D factor for odd totals.
    "factor_types": None,  # Repeats h, s, e; or specify e.g. ["h", "s"] with total 4.
    "curvatures": None,  # Defaults to -1, +1, 0 according to factor types.
    "learn_curvature": True,  # False gives a fixed-curvature product control.
}
PAPER_UNIVERSAL_MODEL = {
    "factor_dims": None,
    "curvature_warmup_epochs": 50,  # First half of the default 100-epoch run.
    "curvature_stabilization_epochs": 5,
    "curvature_epsilon": .001,
}
TRAINING = {
    "epochs": 100,
    "batch_size": 128,
    "learning_rate": 0.001,
    "device": "cpu",
    "curvature_learning_rate": .001,  # SGD for paper model radii/signed curvatures.
}
EVALUATION = {
    "importance_samples": 256,
    "sample_chunk_size": 64,
    "batch_size": 256,
    "seed": 1234,
}
COLOR_VOLUME = False  # Color decoder surfaces by sqrt(det(J.T @ J)).
