"""Euclidean and hyperbolic VAEs with the same training/evaluation interface."""

import math

import torch
from torch import nn
from torch.distributions import Distribution, Independent, Normal


def lorentz_inner(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Minkowski inner product, with the time coordinate first."""
    return (x[..., 1:] * y[..., 1:]).sum(-1) - x[..., 0] * y[..., 0]


def _sinhc(squared: torch.Tensor) -> torch.Tensor:
    """sinh(sqrt(squared))/sqrt(squared), including its derivative at zero."""
    radius = squared.clamp_min(1e-8).sqrt()
    return torch.where(squared < 1e-6, 1 + squared / 6 + squared.square() / 120,
                       radius.sinh() / radius)


def _log_sinhc(squared: torch.Tensor) -> torch.Tensor:
    radius = squared.clamp_min(1e-8).sqrt()
    regular = radius + torch.log(-torch.expm1(-2 * radius)) - math.log(2) - radius.log()
    return torch.where(squared < 1e-6, squared / 6 - squared.square() / 180, regular)


def exp_map_zero(tangent: torch.Tensor, curvature: torch.Tensor) -> torch.Tensor:
    """Map intrinsic origin-tangent coordinates to H^d of curvature < 0."""
    curvature = curvature.to(tangent)
    squared = -curvature * tangent.square().sum(-1, keepdim=True)
    radius = squared.clamp_min(1e-8).sqrt()
    time = torch.where(squared < 1e-6, 1 + squared / 2 + squared.square() / 24,
                       radius.cosh()) / (-curvature).sqrt()
    return torch.cat((time, _sinhc(squared) * tangent), dim=-1)


def log_map_zero(point: torch.Tensor, curvature: torch.Tensor) -> torch.Tensor:
    """Inverse of exp_map_zero; returns d intrinsic tangent coordinates."""
    curvature = curvature.to(point)
    spatial = point[..., 1:]
    squared = -curvature * spatial.square().sum(-1, keepdim=True)
    radius = squared.clamp_min(1e-8).sqrt()
    factor = torch.where(squared < 1e-6, 1 - squared / 6 + 3 * squared.square() / 40,
                         radius.asinh() / radius)
    return factor * spatial


def parallel_transport(
    source: torch.Tensor, target: torch.Tensor, tangent: torch.Tensor,
    curvature: torch.Tensor,
) -> torch.Tensor:
    """Transport a tangent vector along the geodesic from source to target."""
    c = -curvature
    factor = c * lorentz_inner(target, tangent) / (1 - c * lorentz_inner(source, target))
    return tangent + factor.unsqueeze(-1) * (source + target)


class WrappedNormal(Distribution):
    """Hyperbolic wrapped normal, with density relative to Riemannian volume.

    A diagonal Gaussian in the origin tangent space is parallel transported to
    `location`, then exponentiated. The exponential-map volume correction is
    (d - 1) * log(sinh(sqrt(-K) * r) / (sqrt(-K) * r)).
    See Nagano et al. (2019), https://proceedings.mlr.press/v97/nagano19a.html.
    """

    arg_constraints = {}
    has_rsample = True

    def __init__(self, location: torch.Tensor, scale: torch.Tensor, curvature: torch.Tensor):
        if not torch.isfinite(curvature).all() or not (curvature < 0).all():
            raise ValueError("wrapped normal curvature must be finite and negative")
        if location.shape[-1] != scale.shape[-1] + 1:
            raise ValueError("location needs d+1 Lorentz coordinates and scale needs d entries")
        batch_shape = torch.broadcast_shapes(location.shape[:-1], scale.shape[:-1])
        self.location = location.expand(batch_shape + location.shape[-1:])
        self.scale = scale.expand(batch_shape + scale.shape[-1:])
        self.curvature = curvature
        self.base = Independent(Normal(torch.zeros_like(self.scale), self.scale), 1)
        origin_tangent = torch.zeros_like(self.scale)
        self.origin = exp_map_zero(origin_tangent, curvature)
        super().__init__(batch_shape, self.location.shape[-1:], validate_args=False)

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        return self.wrap(self.base.rsample(sample_shape))

    def wrap(self, intrinsic: torch.Tensor) -> torch.Tensor:
        """Transport and exponentiate an origin-tangent vector."""
        tangent = torch.cat((torch.zeros_like(intrinsic[..., :1]), intrinsic), dim=-1)
        transported = parallel_transport(self.origin, self.location, tangent, self.curvature)
        # Transport preserves norm; use the original norm to avoid cancellation.
        squared = -self.curvature * intrinsic.square().sum(-1, keepdim=True)
        radius = squared.clamp_min(1e-8).sqrt()
        cosh = torch.where(squared < 1e-6, 1 + squared / 2 + squared.square() / 24,
                           radius.cosh())
        return cosh * self.location + _sinhc(squared) * transported

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        alpha = (self.curvature * lorentz_inner(self.location, value)).clamp_min(1)
        delta = alpha - 1
        safe_delta = delta.clamp_min(1e-8)
        factor = torch.where(delta < 1e-6, 1 - delta / 3 + 2 * delta.square() / 15,
                             torch.acosh(1 + safe_delta) / (safe_delta * (safe_delta + 2)).sqrt())
        tangent = factor.unsqueeze(-1) * (value - alpha.unsqueeze(-1) * self.location)
        intrinsic = parallel_transport(self.location, self.origin, tangent, self.curvature)[..., 1:]
        squared = -self.curvature * intrinsic.square().sum(-1)
        return self.base.log_prob(intrinsic) - (self.scale.shape[-1] - 1) * _log_sinhc(squared)


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

    def latent_from_tangent(self, tangent: torch.Tensor) -> torch.Tensor:
        """Coordinates used by decoder-surface plots."""
        return tangent


class CurvedVAE(VAE):
    """Hyperbolic VAE with fixed or learned negative sectional curvature.

    latent_dim is the intrinsic dimension; samples have latent_dim + 1 Lorentz
    coordinates. The decoder uses the logarithmic map at the origin, keeping
    the same network architecture as the flat baseline.
    """

    def __init__(
        self, latent_dim: int = 2, encoder_width: int = 64, decoder_width: int = 32,
        observation_std: float = 0.1, curvature: float = -1.0,
        learn_curvature: bool = False,
    ):
        if not math.isfinite(curvature) or curvature >= 0:
            raise ValueError("curvature must be finite and negative; use VAE for flat space")
        super().__init__(latent_dim, encoder_width, decoder_width, observation_std)
        self.learn_curvature = learn_curvature
        if learn_curvature:
            magnitude = -curvature - 1e-6
            if magnitude <= 0:
                raise ValueError("learned curvature must start below -1e-6")
            raw = magnitude + math.log(-math.expm1(-magnitude))
            self.raw_curvature = nn.Parameter(torch.tensor(raw))
        else:
            self.register_buffer("_fixed_curvature", torch.tensor(float(curvature)))

    @property
    def curvature(self) -> torch.Tensor:
        if self.learn_curvature:
            return -(torch.nn.functional.softplus(self.raw_curvature) + 1e-6)
        return self._fixed_curvature

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Preserve checkpoints made before curvature became configurable.
        old_key = prefix + "curvature"
        if not self.learn_curvature and old_key in state_dict:
            state_dict[prefix + "_fixed_curvature"] = state_dict.pop(old_key)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _geometry_dtype(self) -> torch.dtype:
        # Lorentz products subtract large terms. MPS does not support float64.
        return torch.float32 if self.curvature.device.type == "mps" else torch.float64

    def posterior(self, x: torch.Tensor) -> WrappedNormal:
        location, log_variance = self.encoder(x).to(self._geometry_dtype()).chunk(2, dim=-1)
        # Bound tangent scales to avoid exponential overflow in manifold maps.
        scale = (0.5 * log_variance.clamp(-10, 2)).exp()
        return WrappedNormal(self.latent_from_tangent(location), scale, self.curvature)

    def prior(self) -> WrappedNormal:
        tangent = self.prior_location.to(self._geometry_dtype())
        return WrappedNormal(self.latent_from_tangent(tangent), torch.ones_like(tangent), self.curvature)

    def observation(self, z: torch.Tensor) -> Independent:
        tangent = log_map_zero(z.to(self._geometry_dtype()), self.curvature)
        tangent = tangent.to(next(self.decoder.parameters()).dtype)
        return super().observation(tangent)

    def latent_from_tangent(self, tangent: torch.Tensor) -> torch.Tensor:
        return exp_map_zero(tangent.to(self._geometry_dtype()), self.curvature)


def build_model(model_type: str = "flat", **settings) -> VAE:
    if model_type == "flat":
        return VAE(**settings)
    if model_type == "curved":
        return CurvedVAE(**settings)
    raise ValueError(f"unknown model type: {model_type}; expected 'flat' or 'curved'")
