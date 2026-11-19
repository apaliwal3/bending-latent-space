"""Projected constant-curvature wrapped normals from Skopek et al. (2020).

Coordinates are twice the paper's projected coordinates, so the metric at the
origin is I, matching our flat baseline. This is a coordinate change, not a
change in sectional curvature. Spherical densities sum exponential-map inverse
branches, unlike the single-branch density in the reference implementation.
"""

import math

import torch
from torch.distributions import Distribution, Independent, Normal

from geometry import log_at_origin, mobius_add


def exp_origin(vector, curvature):
    """Geodesic exponential of an orthonormal origin-tangent vector."""
    t = curvature * vector.square().sum(-1, keepdim=True) / 4
    h = t.abs().clamp_min(1e-12).sqrt()
    regular = torch.where(curvature < 0, h.tanh() / h, h.tan() / h)
    factor = torch.where(t.abs() < 1e-5, 1 + t / 3 + 2 * t.square() / 15, regular)
    return factor * vector


def log_exp_volume(vector, curvature):
    """Log Riemannian Jacobian of exp: (d-1) log |sin_K(r)/r|."""
    t = curvature * vector.square().sum(-1)
    h = t.abs().clamp_min(1e-12).sqrt()
    negative = h + torch.log(-torch.expm1(-2 * h)) - math.log(2) - h.log()
    positive = h.sin().abs().clamp_min(torch.finfo(vector.dtype).tiny).log() - h.log()
    regular = torch.where(curvature < 0, negative, positive)
    result = torch.where(t.abs() < 1e-5, -t / 6 - t.square() / 180, regular)
    return (vector.shape[-1] - 1) * result


def distance(x, y, curvature):
    """Geodesic distance in a projected factor."""
    return log_at_origin(mobius_add(-x, y, curvature), curvature).norm(dim=-1)


class ProjectedWrappedNormal(Distribution):
    """Gaussian -> parallel transport from origin -> exponential at location.

    In these coordinates transport followed by exp is location (+)_K exp_0(v).
    Density is relative to Riemannian volume. For K>0 an adaptive winding sum
    bounds its omitted density relative to the retained sum by winding_tolerance.
    Failure to resolve the sum raises, rather than silently dropping mass.
    """

    arg_constraints = {}
    has_rsample = True

    def __init__(self, location, scale, curvature, winding_tolerance=1e-10, max_windings=64):
        if curvature.numel() != 1 or not torch.isfinite(curvature).all():
            raise ValueError("curvature must be a finite scalar")
        if not 0 < winding_tolerance < 1 or type(max_windings) is not int or max_windings < 1:
            raise ValueError("invalid winding tolerance or maximum")
        self.location, self.scale = torch.broadcast_tensors(location, scale)
        self.curvature = curvature.to(location)
        self.winding_tolerance, self.max_windings = winding_tolerance, max_windings
        self.base = Independent(Normal(torch.zeros_like(self.scale), self.scale), 1)
        super().__init__(self.location.shape[:-1], self.location.shape[-1:], validate_args=False)

    def wrap(self, vector):
        return mobius_add(self.location, exp_origin(vector, self.curvature), self.curvature)

    def rsample(self, sample_shape=torch.Size()):
        return self.wrap(self.base.rsample(sample_shape))

    def log_prob(self, value):
        centered = mobius_add(-self.location, value, self.curvature)
        vector = log_at_origin(centered, self.curvature)
        principal = self.base.log_prob(vector) - log_exp_volume(vector, self.curvature)
        if self.curvature > 0:
            return self._spherical_log_prob(vector, principal)
        valid = 1 + self.curvature * value.square().sum(-1) / 4 > 0
        return torch.where(valid, principal, -torch.inf)

    def _spherical_log_prob(self, vector, principal):
        r = vector.norm(dim=-1).clamp_min(torch.finfo(vector.dtype).eps)
        direction = vector / r.unsqueeze(-1)
        # Use an arbitrary direction at the singular center. Its density is infinite.
        at_center = vector.square().sum(-1) == 0
        direction = torch.where(at_center.unsqueeze(-1), torch.ones_like(direction) /
                                math.sqrt(self.event_shape[0]), direction)
        root = self.curvature.sqrt()
        period = 2 * math.pi / root
        power = self.event_shape[0] - 1
        log_sin = (root * r).sin().abs().clamp_min(torch.finfo(vector.dtype).tiny).log()
        precision = (direction / self.scale).square().sum(-1)

        def term(signed_radius):
            return (self.base.log_prob(direction * signed_radius.unsqueeze(-1))
                    + power * ((root * signed_radius.abs()).log() - log_sin))

        total = principal
        for n in range(self.max_windings + 1):
            if n:
                total = torch.logaddexp(total, term(r + n * period))
                total = torch.logaddexp(total, term(r - n * period))
            # Gaussian times radius**(d-1) has decreasing ratios past its mode.
            # Bound each omitted half-series by first_term / (1 - next/first).
            with torch.no_grad():
                tails = []
                for first in ((n + 1) * period + r, (n + 1) * period - r):
                    log_ratio = (-precision * (first * period + period.square() / 2)
                                 + power * torch.log1p(period / first))
                    bound = term(first) - torch.log(-torch.expm1(log_ratio.clamp_max(-1e-15)))
                    tails.append(torch.where(log_ratio < 0, bound, torch.inf))
                log_tail = torch.logaddexp(*tails)
                resolved = (log_tail - total <= math.log(self.winding_tolerance)) | at_center
                converged = bool(resolved.all())
            if converged:
                return torch.where(at_center & (power > 0), torch.inf, total)
        raise FloatingPointError("Spherical winding density unresolved; increase max_windings.")


class ProductWrappedNormal(Distribution):
    """Independent factor distributions and concatenated intrinsic events."""

    arg_constraints = {}
    has_rsample = True

    def __init__(self, location, scale, curvatures, factor_dims, **settings):
        self.factor_dims = tuple(factor_dims)
        if (not self.factor_dims or any(type(d) is not int or d < 1 for d in self.factor_dims)
                or sum(self.factor_dims) != location.shape[-1]
                or curvatures.shape != (len(self.factor_dims),)):
            raise ValueError("factor dimensions and curvatures must match the latent event")
        self.location, self.scale = torch.broadcast_tensors(location, scale)
        self.curvatures = curvatures
        self.factors = tuple(ProjectedWrappedNormal(mu, sigma, k, **settings)
                             for mu, sigma, k in zip(self.location.split(self.factor_dims, -1),
                                                      self.scale.split(self.factor_dims, -1), curvatures))
        self.base = Independent(Normal(torch.zeros_like(self.scale), self.scale), 1)
        super().__init__(self.location.shape[:-1], self.location.shape[-1:], validate_args=False)

    def wrap(self, vector):
        return torch.cat([factor.wrap(v) for factor, v in
                          zip(self.factors, vector.split(self.factor_dims, -1))], -1)

    def rsample(self, sample_shape=torch.Size()):
        return self.wrap(self.base.rsample(sample_shape))

    def factor_log_prob(self, value):
        return torch.stack([factor.log_prob(z) for factor, z in
                            zip(self.factors, value.split(self.factor_dims, -1))], -1)

    def log_prob(self, value):
        return self.factor_log_prob(value).sum(-1)
