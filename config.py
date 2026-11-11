"""Edit these settings, then run: uv run python run.py"""

OUTPUT_DIR = "outputs/experiment"

# Every combination runs. Add values here to compare experiments.
DATASETS = ["sphere"]
MODEL_TYPES = ["flat", "curved"]
NOISE_LEVELS = [0.05]
LATENT_DIMS = [2]
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
TRAINING = {
    "epochs": 100,
    "batch_size": 128,
    "learning_rate": 0.001,
    "device": "cpu",
}
EVALUATION = {
    "importance_samples": 256,
    "sample_chunk_size": 64,
    "batch_size": 256,
    "seed": 1234,
}
COLOR_VOLUME = False  # Color decoder surfaces by sqrt(det(J.T @ J)).
