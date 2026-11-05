"""A simple Gaussian VAE. Change the encoder and decoder here."""

import torch
from torch import nn
from torch.distributions import Independent, Normal


class VAE(nn.Module):
    def __init__(
        self,
        latent_dim: int = 2,
        encoder_width: int = 64,
        decoder_width: int = 32,
        observation_std: float = 0.1,
    ):
        super().__init__()
        if min(latent_dim, encoder_width, decoder_width) < 1 or observation_std <= 0:
            raise ValueError("dimensions, widths and observation_std must be positive")
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(3, encoder_width), nn.Tanh(),
            nn.Linear(encoder_width, encoder_width), nn.Tanh(),
            nn.Linear(encoder_width, 2 * latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, decoder_width), nn.Tanh(),
            nn.Linear(decoder_width, decoder_width), nn.Tanh(),
            nn.Linear(decoder_width, 3),
        )
        self.register_buffer("observation_std", torch.tensor(float(observation_std)))
        self.register_buffer("prior_location", torch.zeros(latent_dim))

    def posterior(self, x: torch.Tensor) -> Independent:
        location, log_variance = self.encoder(x).chunk(2, dim=-1)
        scale = (0.5 * log_variance.clamp(-20, 20)).exp()
        return Independent(Normal(location, scale), 1)  # Sum latent event dimensions.

    def prior(self) -> Independent:
        return Independent(Normal(self.prior_location, torch.ones_like(self.prior_location)), 1)

    def observation(self, z: torch.Tensor) -> Independent:
        return Independent(Normal(self.decoder(z), self.observation_std), 1)
