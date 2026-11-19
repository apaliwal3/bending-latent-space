"""Mixed-curvature VAEs following Skopek, Ganea & Becigneul (ICLR 2020).

Implements their diagonal wrapped-normal product model in projected coordinates,
fixed-sign radius learning, and staged universal curvature learning. Networks and
Gaussian observations are adapted to this project's 3D surface experiments.
"""

import math

import torch
from torch import nn
from torch.distributions import Independent, Normal

from model import VAE
from paper_geometry import ProductWrappedNormal, distance, exp_origin


def partition_dimensions(latent_dim):
    """Paper's 2D components, with one 3D component for odd total dimension."""
    if type(latent_dim) is not int or latent_dim < 2:
        raise ValueError("paper curved models require integer latent_dim >= 2")
    return (2,) * (latent_dim // 2 - 1) + (2 + latent_dim % 2,)


class PaperFlatVAE(VAE):
    """Matched flat control: one hidden ReLU layer and softplus posterior scales."""

    def __init__(self, latent_dim=6, encoder_width=64, decoder_width=32, observation_std=.1):
        super().__init__(latent_dim, encoder_width, decoder_width, observation_std)
        self.encoder = nn.Sequential(nn.Linear(3, encoder_width), nn.ReLU(),
                                     nn.Linear(encoder_width, 2 * latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, decoder_width), nn.ReLU(),
                                     nn.Linear(decoder_width, 3))

    def encode_parameters(self, x):
        location, raw_scale = self.encoder(x).chunk(2, -1)
        return location, torch.nn.functional.softplus(raw_scale) + 1e-5

    def posterior(self, x):
        location, scale = self.encode_parameters(x)
        return Independent(Normal(location, scale), 1)


class CurvatureFactor(nn.Module):
    def __init__(self, kind, curvature, learn_curvature):
        super().__init__()
        self.kind = kind
        if not math.isfinite(curvature):
            raise ValueError("initial curvatures must be finite")
        if kind == "u":
            self.signed_curvature = nn.Parameter(torch.tensor(float(curvature)), requires_grad=learn_curvature)
        elif kind in ("h", "s"):
            if curvature == 0 or (curvature < 0) != (kind == "h"):
                raise ValueError("h requires negative curvature; s requires positive curvature")
            self.radius = nn.Parameter(torch.tensor(1 / math.sqrt(abs(curvature))),
                                       requires_grad=learn_curvature)
        elif kind == "e":
            if curvature != 0:
                raise ValueError("Euclidean factors require curvature zero")
            self.register_buffer("zero", torch.tensor(0.))
        else:
            raise ValueError("factor types must be e, h, s or u")

    @property
    def curvature(self):
        if self.kind == "u":
            return self.signed_curvature
        if self.kind == "e":
            return self.zero
        return (-1 if self.kind == "h" else 1) / self.radius.clamp_min(1e-4).square()


class MixedCurvatureVAE(PaperFlatVAE):
    """Product of projected hyperbolic/spherical/Euclidean wrapped normals.

    mode='mixed': fixed signature, with optional SGD learning of factor radii.
    mode='universal': flat warmup, signed epsilon perturbation, stabilization,
    then learning signed curvatures. Assigned flat components remain flat, as in
    the paper. For two factors the perturbed groups are h and s; >=3 includes e.
    """

    def __init__(self, latent_dim=6, encoder_width=64, decoder_width=32, observation_std=.1,
                 factor_dims=None, factor_types=None, curvatures=None, learn_curvature=True,
                 mode="mixed", curvature_warmup_epochs=50, curvature_stabilization_epochs=5,
                 curvature_epsilon=.001, winding_tolerance=1e-10, max_windings=64):
        dims = tuple(factor_dims) if factor_dims is not None else partition_dimensions(latent_dim)
        if not dims or any(type(d) is not int or d < 1 for d in dims) or sum(dims) != latent_dim:
            raise ValueError("factor_dims must be positive integers summing to latent_dim")
        if mode not in ("mixed", "universal"):
            raise ValueError("mode must be mixed or universal")
        if factor_types is None:
            factor_types = ["u"] * len(dims) if mode == "universal" else ["hse"[i % 3] for i in range(len(dims))]
        types = tuple(factor_types)
        if len(types) != len(dims) or any(k not in "ehsu" or len(k) != 1 for k in types):
            raise ValueError("one factor type (e/h/s/u) is required per dimension block")
        if any(d < 2 and k != "e" for d, k in zip(dims, types)):
            raise ValueError("curved factors require dimension >= 2 for sectional curvature")
        if mode == "universal" and (not learn_curvature or any(k != "u" for k in types)):
            raise ValueError("universal mode requires learnable u components")
        if mode == "mixed" and "u" in types:
            raise ValueError("use mode='universal' for sign-changing components")
        if (type(curvature_warmup_epochs) is not int or curvature_warmup_epochs < 0 or
                type(curvature_stabilization_epochs) is not int or curvature_stabilization_epochs < 0 or
                not math.isfinite(curvature_epsilon) or curvature_epsilon <= 0):
            raise ValueError("invalid curvature training schedule")
        initial = tuple(curvatures) if curvatures is not None else tuple(
            -1. if k == "h" else 1. if k == "s" else 0. for k in types)
        if len(initial) != len(dims):
            raise ValueError("one initial curvature is required per factor")
        if mode == "universal" and any(k != 0 for k in initial):
            raise ValueError("staged universal mode must start at zero curvature")
        super().__init__(latent_dim, encoder_width, decoder_width, observation_std)
        self.factor_dims, self.factor_types = dims, types
        self.initial_curvatures = initial
        self.mode, self.learn_curvature = mode, learn_curvature
        self.curvature_warmup_epochs = curvature_warmup_epochs
        self.curvature_stabilization_epochs = curvature_stabilization_epochs
        self.curvature_epsilon = curvature_epsilon
        self.density_settings = dict(winding_tolerance=winding_tolerance, max_windings=max_windings)
        self.factors = nn.ModuleList(CurvatureFactor(k, c, learn_curvature) for k, c in zip(types, initial))
        self.register_buffer("_signature", torch.tensor([[d, "ehsu".index(k)] for d, k in zip(dims, types)]))
        self.register_buffer("_curvature_stage", torch.tensor(0 if mode == "universal" else 2))
        # Round-robin assignment avoids making the two-factor case h + flat.
        self.register_buffer("_universal_signs", torch.tensor([(-1., 1., 0.)[i % 3] for i in range(len(dims))]))
        if mode == "universal":
            for factor, sign in zip(self.factors, self._universal_signs):
                if sign == 0:
                    factor.signed_curvature.requires_grad_(False)

    @property
    def curvatures(self):
        return torch.stack([factor.curvature for factor in self.factors])

    @property
    def curvature_stage(self):
        return ("flat_warmup", "stabilization", "learned")[int(self._curvature_stage)]

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        signature = state_dict.get(prefix + "_signature")
        if signature is not None and not torch.equal(signature.cpu(), self._signature.cpu()):
            raise ValueError("checkpoint factor signature differs from this model")
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def geometry_parameters(self):
        return [p for p in self.factors.parameters() if p.requires_grad]

    def set_training_epoch(self, epoch):
        if self.mode != "universal":
            return
        if epoch <= self.curvature_warmup_epochs:
            stage = 0
        elif epoch <= self.curvature_warmup_epochs + self.curvature_stabilization_epochs:
            stage = 1
        else:
            stage = 2
        if int(self._curvature_stage) == 0 and stage > 0:
            with torch.no_grad():
                for factor, sign in zip(self.factors, self._universal_signs):
                    factor.signed_curvature.copy_(sign * self.curvature_epsilon)
        self._curvature_stage.fill_(stage)

    def prepare_geometry_step(self):
        if self.mode == "universal":
            for factor, sign in zip(self.factors, self._universal_signs):
                if int(self._curvature_stage) != 2 or sign == 0:
                    factor.signed_curvature.grad = None
        parameters = [p for p in self.geometry_parameters() if p.grad is not None]
        if parameters:
            torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)

    def project_geometry_parameters(self):
        with torch.no_grad():
            for factor in self.factors:
                if hasattr(factor, "radius"):
                    factor.radius.clamp_(min=1e-4)

    def _geometry_dtype(self):
        return torch.float32 if self.prior_location.device.type == "mps" else torch.float64

    def latent_from_tangent(self, tangent):
        return torch.cat([exp_origin(v, k) for v, k in
                          zip(tangent.to(self._geometry_dtype()).split(self.factor_dims, -1),
                              self.curvatures)], -1)

    def posterior(self, x):
        location, scale = self.encode_parameters(x)
        return ProductWrappedNormal(self.latent_from_tangent(location), scale.to(self._geometry_dtype()),
                                     self.curvatures, self.factor_dims, **self.density_settings)

    def prior(self):
        location = self.prior_location.to(self._geometry_dtype())
        return ProductWrappedNormal(location, torch.ones_like(location), self.curvatures,
                                     self.factor_dims, **self.density_settings)

    def observation(self, z):
        # Like the reference FFNN, decode concatenated projected points directly.
        return super().observation(z.to(next(self.decoder.parameters()).dtype))

    def factor_distances(self, x, y):
        return torch.stack([distance(a, b, k) for a, b, k in
                            zip(x.split(self.factor_dims, -1), y.split(self.factor_dims, -1),
                                self.curvatures)], -1)
