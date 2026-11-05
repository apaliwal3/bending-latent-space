"""Sphere, Swiss roll and saddle data with fixed train/validation/test splits."""

import json
from pathlib import Path

import numpy as np


def sphere(n: int, rng: np.random.Generator, radius: float = 1.0):
    if radius <= 0:
        raise ValueError("radius must be positive")
    directions = rng.normal(size=(n, 3))
    points = radius * directions / np.linalg.norm(directions, axis=1, keepdims=True)
    return points, points / radius, np.full(n, 1 / radius**2), {"radius": radius}


def swiss_roll(
    n: int,
    rng: np.random.Generator,
    t_min: float = 1.5 * np.pi,
    t_max: float = 4.5 * np.pi,
    height: float = 2.0,
    radius_scale: float = 0.1,
):
    if t_max <= t_min or height <= 0 or radius_scale <= 0:
        raise ValueError("require t_max > t_min, height > 0 and radius_scale > 0")
    t = rng.uniform(t_min, t_max, n)
    h = rng.uniform(-height / 2, height / 2, n)
    points = np.column_stack((radius_scale * t * np.cos(t), h, radius_scale * t * np.sin(t)))
    parameters = {"t_min": t_min, "t_max": t_max, "height": height, "radius_scale": radius_scale}
    return points, np.column_stack((t, h)), np.zeros(n), parameters


def saddle(
    n: int, rng: np.random.Generator, extent: float = 1.0, a: float = 0.5, b: float = 0.5
):
    if extent <= 0 or a <= 0 or b <= 0:
        raise ValueError("extent, a and b must be positive")
    u, v = rng.uniform(-extent, extent, (2, n))
    points = np.column_stack((u, v, a * u**2 - b * v**2))
    curvature = -4 * a * b / (1 + 4 * a**2 * u**2 + 4 * b**2 * v**2) ** 2
    return points, np.column_stack((u, v)), curvature, {"extent": extent, "a": a, "b": b}


def generate_data(config: dict) -> dict:
    n = config["n_samples"]
    train_fraction = config.get("train_fraction", 0.7)
    val_fraction = config.get("val_fraction", 0.15)
    noise_std = config["noise_std"]
    if n < 3 or noise_std < 0:
        raise ValueError("require at least three samples and nonnegative noise")
    if not (0 < train_fraction < 1 and 0 < val_fraction < 1 - train_fraction):
        raise ValueError("train/validation fractions must leave a nonempty test fraction")
    n_train, n_val = int(n * train_fraction), int(n * val_fraction)
    if min(n_train, n_val, n - n_train - n_val) < 1:
        raise ValueError("each split needs at least one point")

    surface_seed, noise_seed = np.random.SeedSequence(config["seed"]).spawn(2)
    rng = np.random.default_rng(surface_seed)
    name = config["name"]
    if name == "sphere":
        clean, coordinates, curvature, parameters = sphere(n, rng, **config.get("parameters", {}))
        sampling = "uniform surface area"
    elif name == "swiss_roll":
        clean, coordinates, curvature, parameters = swiss_roll(n, rng, **config.get("parameters", {}))
        sampling = "uniform t and height; not uniform surface area"
    elif name == "saddle":
        clean, coordinates, curvature, parameters = saddle(n, rng, **config.get("parameters", {}))
        sampling = "uniform u and v; not uniform surface area"
    else:
        raise ValueError(f"unknown dataset: {name}")
    noise = np.random.default_rng(noise_seed).normal(size=(n, 3))
    points = (clean + noise_std * noise).astype(np.float32)
    order = np.random.default_rng(config["split_seed"]).permutation(n)
    metadata = {
        "config": config,
        "parameters": parameters,
        "sampling": sampling,
        "curvature_definition": "Gaussian curvature of the clean surface at each point",
        "constant_curvature": float(curvature[0]) if name != "saddle" else None,
        "curvature_min": float(curvature.min()),
        "curvature_max": float(curvature.max()),
    }
    return {
        "points": points,
        "clean": clean.astype(np.float32),
        "coordinates": coordinates,
        "curvature": curvature,
        "train_idx": order[:n_train],
        "val_idx": order[n_train : n_train + n_val],
        "test_idx": order[n_train + n_val :],
        "metadata": metadata,
    }


def save_data(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: value for key, value in data.items() if key != "metadata"}
    np.savez_compressed(path, **arrays, metadata=json.dumps(data["metadata"]))


def load_data(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files if key != "metadata"}
        data["metadata"] = json.loads(str(archive["metadata"]))
    return data


def split_points(data: dict, split: str) -> np.ndarray:
    return data["points"][data[f"{split}_idx"]]
