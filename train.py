"""ELBO training and checkpoints. A later curved model uses the same interface."""

import json
from pathlib import Path

import numpy as np
import torch


def elbo_loss(model, x: torch.Tensor) -> torch.Tensor:
    q = model.posterior(x)
    z = q.rsample()
    log_likelihood = model.observation(z).log_prob(x)
    kl_sample = q.log_prob(z) - model.prior().log_prob(z)  # Same latent reference measure.
    return (kl_sample - log_likelihood).mean()


def rng_state(device: str) -> dict:
    state = {"cpu": torch.get_rng_state()}
    if torch.device(device).type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    elif torch.device(device).type == "mps":
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng(state: dict) -> None:
    torch.set_rng_state(state["cpu"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state:
        torch.mps.set_rng_state(state["mps"])


def train(
    model, train_points: np.ndarray, val_points: np.ndarray, output_dir: str | Path,
    epochs: int = 100, batch_size: int = 128, learning_rate: float = 0.001,
    device: str = "cpu", seed: int = 0,
) -> list[dict]:
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch size and learning rate must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history, start_epoch, best_loss = [], 0, float("inf")
    if (output_dir / "last.pt").exists():
        saved = torch.load(output_dir / "last.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        history, start_epoch, best_loss = saved["history"], saved["epoch"], saved["best_loss"]
        restore_rng(saved["rng"])
    if epochs < start_epoch:
        raise ValueError(f"epochs must be at least the saved epoch ({start_epoch})")
    train_points = torch.as_tensor(train_points)
    val_points = torch.as_tensor(val_points)
    for epoch in range(start_epoch + 1, epochs + 1):
        model.train()
        train_loss = 0.0
        for indices in torch.randperm(len(train_points)).split(batch_size):
            x = train_points[indices].to(device)
            optimizer.zero_grad()
            loss = elbo_loss(model, x)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach()) * len(x)
        model.eval()
        state = rng_state(device)
        torch.manual_seed(seed + 10000)  # Fixed validation samples across epochs.
        val_loss = 0.0
        with torch.no_grad():
            for x in val_points.split(batch_size):
                val_loss += float(elbo_loss(model, x.to(device))) * len(x)
        restore_rng(state)
        train_loss /= len(train_points)
        val_loss /= len(val_points)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        improved = val_loss < best_loss
        best_loss = min(best_loss, val_loss)
        saved = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "epoch": epoch, "best_loss": best_loss, "history": history, "rng": rng_state(device),
        }
        if improved:
            torch.save(saved, output_dir / "best.pt")
        torch.save(saved, output_dir / "last.pt")
        (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        if epoch == 1 or epoch == epochs or epoch % 10 == 0:
            print(f"Epoch {epoch}: train loss {train_loss:.3f}, validation loss {val_loss:.3f}", flush=True)
    return history
