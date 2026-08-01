"""LeJEPA loss: JEPA prediction loss + SIGReg 
this module reimplements the paper's logic directly rather than importing their package, &
drops their distributed (`all_reduce`) handling since this repo targets a
single GPU
"""
import torch
import torch.nn as nn


class SIGReg(nn.Module):
    """Epps-Pulley goodness-of-fit statistic, applied over random 1-D
    projections of the embeddings, pushing their distribution toward an
    isotropic standard Gaussian.

    One deviation fromthe paper's Algorithm 1: instead
    of integrating t over a symmetric range like linspace(-5, 5, 17), this
    integrates only t in [0, t_max] with `num_knots=17` trapezoid points,
    then doubles via symmetry (both the empirical & the standard-normal
    characteristic functions are even/odd in exactly the way that makes the
    integrand over [-t_max, 0] and [0, t_max] identical), same quadrature
    result, 1/2 the compute. num_knots=17,
    `EppsPulley`/`SIGReg` classes; t_max=5 matches the paper's own
    Algorithm 1 pseudocode
    gitcode instead hardcodes t_max=3,
    which the paper calls "negligible impact" either way)
    """

    def __init__(self, num_slices=256, t_max=5.0, num_knots=17):
        super().__init__()
        self.num_slices = num_slices

        t = torch.linspace(0.0, t_max, num_knots)
        dt = t_max / (num_knots - 1)
        quad_weights = torch.full((num_knots,), 2 * dt)
        quad_weights[[0, -1]] = dt  # trapezoid rule 1/2 weight at the endpoints
        phi = torch.exp(-t.square() / 2)  # standard normal char. function (real part; imag part is 0)

        self.register_buffer('t', t)
        self.register_buffer('phi', phi)
        # phi doubles as the Epps-Pulley integration window not just the CF target
        self.register_buffer('weights', quad_weights * phi)

    def forward(self, z):
        """z: (..., N, D) embeddings, N = batch size (samples), D = proj_dim.
        Leading dims (e.g. a view axis) are preserved and averaged over at
        the end, matching the reference `SIGReg.forward`'s `(V, N, D)` usage.
        Random directions are drawn fresh every call, as required for small
        `num_slices` to work well (paper Sec. 4)."""
        n, d = z.shape[-2], z.shape[-1]
        directions = torch.randn(d, self.num_slices, device=z.device, dtype=z.dtype)
        directions = directions / directions.norm(p=2, dim=0, keepdim=True)  # unit directions

        proj = z @ directions  # (..., N, M) scalar projections onto each random direction
        angles = proj.unsqueeze(-1) * self.t  # (..., N, M, K)

        ecf_cos = angles.cos().mean(-3)  # empirical CF, real part, averaged over the N samples
        ecf_sin = angles.sin().mean(-3)  # empirical CF, imaginary part
        squared_err = (ecf_cos - self.phi).square() + ecf_sin.square()  # (..., M, K)

        statistic = (squared_err @ self.weights) * n  # trapezoid integral over t, Epps-Pulley n-scaling
        return statistic.mean()  # average over slices


def jepa_prediction_loss(proj):
    """proj: (V, N, D) projected embeddings for V views of N samples.
    L_pred = mean_v || mu_n - z_{n,v} ||^2 (Eq. 5-7): each view should
    predict the shared /sample mean embedding across its views
    no separate predictor network needed"""
    mean_embedding = proj.mean(dim=0, keepdim=True)  # (1, N, D), the shared target mu_n
    return (proj - mean_embedding).square().mean()


class LeJEPALoss(nn.Module):
    """Total loss L = (1 - lambda) * L_pred + lambda * SIGReg (Eq. 8)"""

    def __init__(self, lam=0.05, num_slices=256):
        super().__init__()
        self.lam = lam
        self.sigreg = SIGReg(num_slices=num_slices)

    def forward(self, proj):
        """proj: (V, N, D) projected embeddings from encoder
        Returns (total_loss, pred_loss, sigreg_loss) so callers can log all
        3 components separately."""
        pred_loss = jepa_prediction_loss(proj)
        sigreg_loss = self.sigreg(proj)
        total_loss = (1 - self.lam) * pred_loss + self.lam * sigreg_loss
        return total_loss, pred_loss, sigreg_loss
