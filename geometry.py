"""Signed stereographic geometry and a normalized Gaussian pushforward family.

Metric: g_z = (1 + K ||z||^2 / 4)^(-2) I, of sectional curvature K.
K < 0: open ball of radius 2/sqrt(-K). K > 0: a sphere minus one pole.
The radial chart transform below is bijective and C1 in K at zero; it is
not the exponential-map wrapped normal used by the Lorentz baseline.
"""

import torch
from torch.distributions import Distribution, Independent, Normal


def mobius_add(x, y, curvature):
    k = curvature / 4
    a = 1 - k * (x * y).sum(-1, keepdim=True)
    # Sum-of-squares form avoids cancellation near a spherical chart pole.
    # W_ij = x_i*y_j - y_i*x_j; ||W||_F^2 / 2 = ||x wedge y||^2.
    wedge = k * (x.unsqueeze(-1) * y.unsqueeze(-2) - y.unsqueeze(-1) * x.unsqueeze(-2))
    numerator = a * (x + y) - (wedge * (x + y).unsqueeze(-2)).sum(-1)
    denominator = a.square() + wedge.square().sum((-2, -1)).unsqueeze(-1) / 2
    # The exact pole remains undefined; do not hide it with a tiny denominator.
    return numerator / denominator


def radial_map(vector, curvature):
    """Biject R^d onto the chart domain, with value/derivative continuous at K=0."""
    r2 = vector.square().sum(-1, keepdim=True)
    negative = 2 / (1 + (1 + (-curvature).clamp_min(0) * r2).sqrt())
    positive = 1 + curvature * r2 / 4
    return vector * torch.where(curvature < 0, negative, positive)


def radial_inverse(point, curvature):
    s2 = point.square().sum(-1, keepdim=True)
    negative = (1 + curvature * s2 / 4).clamp_min(torch.finfo(point.dtype).eps).reciprocal()
    # Solve r + K*r^3/4 = s, using the hyperbolic closed form of the cubic.
    a = curvature * s2 / 4
    h = (3 * a.clamp_min(0)).clamp_min(1e-10).sqrt()
    regular = 2 * torch.sinh(torch.asinh(1.5 * h) / 3) / h
    positive = torch.where(a.abs() < 1e-6, 1 - a + 3 * a.square(), regular)
    return point * torch.where(curvature < 0, negative, positive)


def radial_logdet(vector, point, curvature):
    """Euclidean log determinant of radial_map at vector."""
    dimension = vector.shape[-1]
    t = curvature * point.square().sum(-1) / 4
    eps = torch.finfo(vector.dtype).eps
    negative = (dimension + 1) * (1 + t).clamp_min(eps).log() - (1 - t).clamp_min(eps).log()
    a = curvature * vector.square().sum(-1) / 4
    positive = (dimension - 1) * (1 + a).clamp_min(eps).log() + (1 + 3 * a).clamp_min(eps).log()
    return torch.where(curvature < 0, negative, positive)


def log_at_origin(point, curvature):
    """Geodesic normal coordinates, with an analytic Euclidean limit."""
    t = curvature * point.square().sum(-1, keepdim=True) / 4
    h = t.abs().clamp_min(1e-10).sqrt()
    eps = torch.finfo(point.dtype).eps
    negative = torch.atanh(h.clamp_max(1 - eps)) / h
    positive = torch.atan(h) / h
    factor = torch.where(curvature < 0, negative, positive)
    factor = torch.where(t.abs() < 1e-6, 1 - t / 3 + t.square() / 5, factor)
    return factor * point


class StereographicNormal(Distribution):
    """Radial Gaussian pushforward followed by an isometric chart translation.

    Both prior and posterior have full manifold support up to null sets. Densities
    are relative to Riemannian volume. Unlike spherical exponential wrapping,
    the radial transform is one-to-one, so no omitted winding terms arise.
    """

    arg_constraints = {}
    has_rsample = True

    def __init__(self, location, scale, curvature):
        self.location, self.scale = torch.broadcast_tensors(location, scale)
        self.curvature = curvature.to(location)
        self.base = Independent(Normal(torch.zeros_like(self.scale), self.scale), 1)
        super().__init__(self.location.shape[:-1], self.location.shape[-1:], validate_args=False)

    def wrap(self, vector):
        return mobius_add(self.location, radial_map(vector, self.curvature), self.curvature)

    def rsample(self, sample_shape=torch.Size()):
        return self.wrap(self.base.rsample(sample_shape))

    def log_prob(self, value):
        centered = mobius_add(-self.location, value, self.curvature)
        vector = radial_inverse(centered, self.curvature)
        conformal = 1 + self.curvature * centered.square().sum(-1) / 4
        log_volume = -self.event_shape[0] * conformal.clamp_min(torch.finfo(value.dtype).eps).log()
        density = self.base.log_prob(vector) - radial_logdet(vector, centered, self.curvature) - log_volume
        valid = 1 + self.curvature * value.square().sum(-1) / 4 > 0
        return torch.where(valid, density, -torch.inf)
