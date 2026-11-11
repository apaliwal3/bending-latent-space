"""Run the experiments listed in config.py, including evaluation and plots."""

from copy import deepcopy
import csv
from itertools import product
import json
from pathlib import Path

import numpy as np
import torch

import config
from data import generate_data, load_data, save_data, split_points
from evaluate import evaluate
from model import build_model
from plot import plot_decoder_surface, plot_history, plot_iwll, plot_points
from train import train


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run_experiment(settings: dict, output_dir: Path, color_volume: bool = False) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    settings_path = output_dir / "config.json"
    if settings_path.exists():
        previous = json.loads(settings_path.read_text())
        previous["training"]["epochs"] = settings["training"]["epochs"]
        # Older flat runs predate the model selector.
        previous["model"].setdefault("type", "flat")
        current = deepcopy(settings)
        current["model"].setdefault("type", "flat")
        for value in (previous, current):
            if value["model"]["type"] == "curved":
                value["model"].setdefault("learn_curvature", False)
        if previous != current:
            raise ValueError("Saved settings differ. Choose another OUTPUT_DIR in config.py.")
    if (output_dir / "last.pt").exists():
        saved = torch.load(output_dir / "last.pt", map_location="cpu", weights_only=True)
        if settings["training"]["epochs"] < saved["epoch"]:
            raise ValueError("Epoch count is smaller than the saved epoch. Choose another OUTPUT_DIR.")
    save_json(settings_path, settings)
    if (output_dir / "data.npz").exists():
        data = load_data(output_dir / "data.npz")
    else:
        data = generate_data(settings["data"])
        save_data(data, output_dir / "data.npz")
    print(f"\n{output_dir}", flush=True)
    torch.manual_seed(settings["training"]["seed"])
    model_settings = dict(settings["model"])
    model = build_model(model_settings.pop("type", "flat"), **model_settings)
    history = train(model, split_points(data, "train"), split_points(data, "val"),
                    output_dir, **settings["training"])
    best = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(best["model"])
    metrics, per_point = evaluate(model, split_points(data, "test"),
                                  device=settings["training"]["device"], **settings["evaluation"])
    result = {
        "dataset": settings["data"]["name"], "noise_std": settings["data"]["noise_std"],
        "latent_dim": settings["model"]["latent_dim"], "decoder_width": settings["model"]["decoder_width"],
        "seed": settings["training"]["seed"], "model_type": type(model).__name__,
        "curvature": float(model.curvature.detach()) if hasattr(model, "curvature") else 0.0,
        "initial_curvature": settings["model"].get("curvature", 0.0),
        "learn_curvature": settings["model"].get("learn_curvature", False),
        "best_epoch": best["epoch"], "n_parameters": sum(p.numel() for p in model.parameters()),
        **metrics,
    }
    save_json(output_dir / "results.json", result)
    np.savez_compressed(output_dir / "test_metrics.npz", **per_point, indices=data["test_idx"])
    torch.manual_seed(settings["evaluation"]["seed"])
    with torch.no_grad():
        generated = model.observation(model.prior().sample((1000,))).sample().cpu().numpy()
    np.savez_compressed(output_dir / "generated.npz", points=generated)
    plot_points(data["points"], output_dir / "data.png")
    plot_points(split_points(data, "test"), output_dir / "point_clouds.png", generated)
    plot_history(history, output_dir / "loss.png")
    if model.latent_dim == 2:
        plot_decoder_surface(model, output_dir / "decoder_surface.png", color_volume=color_volume)
    print(f"Test IWLL: {metrics['iwll']:.3f} nats/point", flush=True)
    return result


def summarize(results: list[dict], output_dir: Path) -> list[dict]:
    keys = ("model_type", "initial_curvature", "learn_curvature", "dataset", "noise_std",
            "latent_dim", "decoder_width")
    groups = {}
    for result in results:
        key = tuple(result[name] for name in keys)
        groups.setdefault(key, []).append(result)
    summary = []
    for key, rows in groups.items():
        row = dict(zip(keys, key))
        row.update(n_seeds=len(rows), seeds=[result["seed"] for result in rows])
        for metric in ("iwll", "elbo", "reconstruction_mse", "reconstruction_nll", "kl", "curvature"):
            values = [result[metric] for result in rows]
            row[f"{metric}_values"] = values
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        summary.append(row)
    save_json(output_dir / "summary.json", summary)
    with (output_dir / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    plot_iwll(summary, output_dir / "iwll.png")
    return summary


def main():
    output_dir = Path(config.OUTPUT_DIR)
    results = []
    combinations = product(config.MODEL_TYPES, config.DATASETS, config.NOISE_LEVELS, config.LATENT_DIMS,
                           config.DECODER_WIDTHS, config.SEEDS)
    for model_type, dataset, noise, latent_dim, width, seed in combinations:
        model_settings = {**config.MODEL, "type": model_type,
                          "latent_dim": latent_dim, "decoder_width": width}
        if model_type == "curved":
            model_settings.update(config.CURVED_MODEL)
        settings = {
            "data": {**config.DATA, "name": dataset, "noise_std": noise,
                     "parameters": config.SURFACES[dataset]},
            "model": model_settings,
            "training": {**config.TRAINING, "seed": seed},
            "evaluation": deepcopy(config.EVALUATION),
        }
        name = f"{dataset}_noise{noise:g}_z{latent_dim}_width{width}_seed{seed}"
        # Retain existing flat experiment paths/checkpoints.
        if model_type != "flat":
            name = f"{model_type}_k{model_settings['curvature']:g}_{name}"
            if model_settings.get("learn_curvature", False):
                name = f"learned_{name}"
        results.append(run_experiment(settings, output_dir / name, config.COLOR_VOLUME))
    summarize(results, output_dir)
    print(f"\nResults and plots saved in {output_dir}")


if __name__ == "__main__":
    main()
