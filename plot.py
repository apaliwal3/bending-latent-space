"""Saved Matplotlib figures; functions also work from notebooks."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colormaps
from matplotlib.colors import Normalize


def finish(fig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_points(observed: np.ndarray, path: str | Path, generated: np.ndarray | None = None):
    clouds = [("Observed", observed)]
    if generated is not None:
        clouds.append(("Prior predictive (includes observation noise)", generated))
    all_points = np.concatenate([points[:2000] for _, points in clouds])
    center = (all_points.min(axis=0) + all_points.max(axis=0)) / 2
    half_range = max(float(np.ptp(all_points, axis=0).max()) / 2, 1e-6)
    fig = plt.figure(figsize=(6 * len(clouds), 5))
    for index, (title, points) in enumerate(clouds, 1):
        ax = fig.add_subplot(1, len(clouds), index, projection="3d")
        points = points[:2000]
        ax.scatter(*points.T, c=points[:, 2], s=5, alpha=0.6, cmap="viridis")
        ax.set(xlabel="x", ylabel="y", zlabel="z", title=title)
        ax.set_xlim(center[0] - half_range, center[0] + half_range)
        ax.set_ylim(center[1] - half_range, center[1] + half_range)
        ax.set_zlim(center[2] - half_range, center[2] + half_range)
        ax.set_box_aspect((1, 1, 1))
    finish(fig, path)


def plot_history(history: list[dict], path: str | Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, label in (("train_loss", "Training"), ("val_loss", "Validation")):
        ax.plot([row["epoch"] for row in history], [row[key] for row in history], label=label)
    ax.set(xlabel="Epoch", ylabel="Negative ELBO (nats/point)")
    ax.legend()
    finish(fig, path)


def plot_iwll(groups: list[dict], path: str | Path):
    """One panel per dataset/noise level, with a line for each decoder width."""
    panels = sorted({(row["dataset"], row["noise_std"]) for row in groups})
    if not panels:
        return
    fig, axes = plt.subplots(len(panels), 1, figsize=(8, 4 * len(panels)), squeeze=False)
    for ax, (dataset, noise) in zip(axes[:, 0], panels):
        subset = [row for row in groups if row["dataset"] == dataset and row["noise_std"] == noise]
        widths = sorted({row["decoder_width"] for row in subset})
        for width in widths:
            rows = sorted((row for row in subset if row["decoder_width"] == width),
                          key=lambda row: row["latent_dim"])
            x = [row["latent_dim"] for row in rows]
            y = [row["iwll_mean"] for row in rows]
            std = [np.nan if row["iwll_std"] is None else row["iwll_std"] for row in rows]
            line, = ax.plot(x, y, "o-", label=f"Decoder width {width}")
            ax.errorbar(x, y, yerr=std, fmt="none", color=line.get_color(), capsize=3)
            for row in rows:
                ax.scatter([row["latent_dim"]] * row["n_seeds"], row["iwll_values"],
                           color=line.get_color(), alpha=0.4, s=16)
        ax.set(title=f"{dataset}, noise std={noise}", xlabel="Latent dimension",
               ylabel="Test IWLL (nats/point; higher is better)")
        ax.set_xticks(sorted({row["latent_dim"] for row in subset}))
        ax.legend(fontsize="small")
    finish(fig, path)


def plot_curvature_recovery(learned, truth, path: str | Path):
    """Compare scalar learned Gaussian curvatures with meaningful scalar targets."""
    learned, truth = np.broadcast_arrays(np.asarray(learned), np.asarray(truth))
    low, high = min(learned.min(), truth.min()), max(learned.max(), truth.max())
    margin = max((high - low) * 0.1, 0.1)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(truth, learned, alpha=0.7)
    ax.plot([low - margin, high + margin], [low - margin, high + margin], "k--", linewidth=1)
    ax.set(xlabel="Ground-truth Gaussian curvature", ylabel="Learned Gaussian curvature")
    finish(fig, path)


def plot_decoder_surface(
    model, path: str | Path, grid_min: float = -2.0, grid_max: float = 2.0,
    resolution: int = 35, color_volume: bool = False,
):
    """Plot f(z); optional color is sqrt(det(J_f.T @ J_f)) in grid coordinates."""
    if model.latent_dim != 2:
        raise ValueError("decoder surface visualization requires latent_dim=2")
    if resolution < 2 or grid_max <= grid_min:
        raise ValueError("require resolution >= 2 and grid_max > grid_min")
    was_training = model.training
    model.eval()
    parameter = next(model.parameters())
    axis = torch.linspace(grid_min, grid_max, resolution, device=parameter.device, dtype=parameter.dtype)
    u, v = torch.meshgrid(axis, axis, indexing="ij")
    z = torch.stack((u, v), dim=-1).reshape(-1, 2)
    with torch.no_grad():
        points = model.observation(z).mean.cpu().numpy().reshape(resolution, resolution, 3)
    surface_options = {"cmap": "viridis"}
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    if color_volume:
        volumes = []
        with torch.enable_grad():
            for latent in z:
                jacobian = torch.autograd.functional.jacobian(lambda value: model.observation(value).mean, latent)
                gram = jacobian.T @ jacobian
                volumes.append(float(torch.linalg.det(gram).clamp_min(0).sqrt()))
        volumes = np.asarray(volumes).reshape(resolution, resolution)
        norm = Normalize(vmin=volumes.min(), vmax=max(volumes.max(), volumes.min() + 1e-12))
        surface_options = {"facecolors": colormaps["viridis"](norm(volumes)), "shade": False}
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax,
                     shrink=0.6, label="Pullback volume element in grid coordinates")
    ax.plot_surface(points[:, :, 0], points[:, :, 1], points[:, :, 2], **surface_options)
    ax.set(xlabel="x", ylabel="y", zlabel="z", title="Decoder mean on a 2D latent grid")
    ax.set_box_aspect(np.maximum(np.ptp(points, axis=(0, 1)), 1e-6))
    finish(fig, path)
    model.train(was_training)
