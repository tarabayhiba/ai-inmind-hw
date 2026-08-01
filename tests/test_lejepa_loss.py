"""Correctness checks for lejepa_loss.py.

Not a pytest suite (this repo has no test framework dependency, SIGReg &
jepa_prediction_loss are pure math with no data/GPU dependency, so it's cheap
to check they actually do what the paper claims, not just that they run
without crashing:

  - SIGReg should score a true standard Gaussian low, & score anything
    that isn't one (collapsed/degenerate, wrong-variance, non-Gaussian
    marginal) meaningfully higher. that's the entire point of the
    statistic (it's what prevents representation collapse in LeJEPA).
  - Gradient descent directly against the SIGReg loss should actually pull a
    non-Gaussian distribution toward standard-Gaussian statistics, this
    catches sign/formula bugs that produce a "valid-looking" but wrong
    gradient (an all-ones-shape smoke test can't catch this)
  - jepa_prediction_loss should be exactly 0 when all views agree, and grow
    with view disagreement.
  - Random projection directions must be resampled every call (Sec.
    4's stated reason small num_slices works), not fixed at construction.

"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from lejepa_loss import LeJEPALoss, SIGReg, jepa_prediction_loss

torch.manual_seed(0)

_all_ok = True


def check(name, cond, detail=""):
    global _all_ok
    _all_ok = _all_ok and bool(cond)
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))


def _avg_score(sigreg, sampler, trials=10):
    # A single draw is noisy (both the data sample and SIGReg's own
    # resampled projection directions add variance) -- average several
    # independent draws for a stable comparison between distributions.
    vals = [sigreg(sampler()).item() for _ in range(trials)]
    return sum(vals) / len(vals)


def test_sigreg_distinguishes_gaussian_from_non_gaussian():
    N, D = 4096, 32
    sigreg = SIGReg(num_slices=256)

    gaussian_score = _avg_score(sigreg, lambda: torch.randn(N, D))
    # Empirical null baseline for this (N, D, num_slices, quadrature) config
    # is ~1-1.3, not ~0 -- the Epps-Pulley statistic's null distribution has
    # a nonzero mean, it isn't a "distance from zero" metric. Sanity-anchor
    # it to a wide-but-bounded range rather than asserting near-zero.
    check(
        "true standard Gaussian scores near its own null baseline",
        0.3 < gaussian_score < 3.0,
        f"score={gaussian_score:.4f}",
    )

    # Full collapse: every embedding is the same point. This is the exact
    # failure mode SIGReg exists to catch (a degenerate delta distribution
    # is about as far from an isotropic Gaussian as you can get).
    collapsed_score = _avg_score(sigreg, lambda: torch.zeros(N, D) + torch.randn(1, D))
    check(
        "collapsed embeddings score far higher than Gaussian",
        collapsed_score > 10 * gaussian_score,
        f"collapsed={collapsed_score:.4f} vs gaussian={gaussian_score:.4f}",
    )

    # Right shape, wrong variance: SIGReg targets a *standard* (unit
    # variance) Gaussian specifically, not any Gaussian.
    scaled_score = _avg_score(sigreg, lambda: torch.randn(N, D) * 5.0)
    check(
        "wrong-variance Gaussian scores higher than standard Gaussian",
        scaled_score > 2 * gaussian_score,
        f"scaled={scaled_score:.4f} vs gaussian={gaussian_score:.4f}",
    )

    # Non-Gaussian marginal, but zero-mean/unit-variance like the target --
    # only the shape of the distribution differs. Uniform is only mildly
    # non-Gaussian (light tails, similar low moments) so the separation is
    # small -- averaging is what makes this check non-flaky.
    uniform_score = _avg_score(sigreg, lambda: (torch.rand(N, D) - 0.5) * math.sqrt(12.0))
    check(
        "unit-variance uniform scores higher than standard Gaussian",
        uniform_score > 1.05 * gaussian_score,
        f"uniform={uniform_score:.4f} vs gaussian={gaussian_score:.4f}",
    )

    # A strongly bimodal (two-point-ish) unit-variance distribution is a
    # much bigger departure from Gaussian than uniform -- should give a
    # large, easily separable score, similar in spirit to the collapse case.
    bimodal_score = _avg_score(
        sigreg, lambda: (torch.randint(0, 2, (N, D)).float() * 2 - 1) * 1.2 + 0.3 * torch.randn(N, D)
    )
    check(
        "strongly bimodal unit-variance distribution scores far higher than Gaussian",
        bimodal_score > 10 * gaussian_score,
        f"bimodal={bimodal_score:.4f} vs gaussian={gaussian_score:.4f}",
    )


def test_sigreg_resamples_directions_every_call():
    sigreg = SIGReg(num_slices=256)
    z = torch.randn(512, 16)
    a = sigreg(z).item()
    b = sigreg(z).item()
    check(
        "two calls on the same input give different scores (fresh random directions)",
        a != b,
        f"call1={a:.6f} call2={b:.6f}",
    )


def test_sigreg_gradient_pulls_toward_gaussian():
    # Start from an intentionally bad (shifted, scaled, skewed) distribution
    # and optimize *only* the SIGReg loss via gradient descent on the raw
    # samples. If the gradient direction were wrong, this either wouldn't
    # converge or would drift away from standard-normal statistics.
    torch.manual_seed(0)
    N, D = 2048, 16
    z = (torch.rand(N, D) ** 2) * 6.0 - 1.0  # skewed, non-zero mean, non-unit variance
    z.requires_grad_(True)

    sigreg = SIGReg(num_slices=256)
    initial_score = sigreg(z).item()

    opt = torch.optim.Adam([z], lr=0.05)
    for _ in range(300):
        opt.zero_grad()
        loss = sigreg(z)
        loss.backward()
        opt.step()

    with torch.no_grad():
        final_score = sigreg(z).item()
        final_mean = z.mean().item()
        final_std = z.std().item()

    check(
        "SIGReg loss drops substantially under gradient descent",
        final_score < 0.3 * initial_score,
        f"initial={initial_score:.4f} final={final_score:.4f}",
    )
    check(
        "optimized distribution's mean moves toward 0",
        abs(final_mean) < 0.2,
        f"final_mean={final_mean:.4f}",
    )
    check(
        "optimized distribution's std moves toward 1",
        0.7 < final_std < 1.3,
        f"final_std={final_std:.4f}",
    )


def test_jepa_prediction_loss():
    V, N, D = 4, 64, 32

    identical = torch.randn(1, N, D).expand(V, N, D)
    loss_identical = jepa_prediction_loss(identical).item()
    check("identical views give exactly zero prediction loss", loss_identical == 0.0, f"loss={loss_identical}")

    low_spread = torch.randn(1, N, D).expand(V, N, D) + 0.1 * torch.randn(V, N, D)
    high_spread = torch.randn(1, N, D).expand(V, N, D) + 1.0 * torch.randn(V, N, D)
    loss_low = jepa_prediction_loss(low_spread).item()
    loss_high = jepa_prediction_loss(high_spread).item()
    check(
        "prediction loss grows with view disagreement",
        loss_high > loss_low,
        f"low_spread={loss_low:.4f} high_spread={loss_high:.4f}",
    )


def test_lejepa_loss_combines_correctly():
    V, N, D = 4, 128, 64
    lam = 0.05
    proj = torch.randn(V, N, D, requires_grad=True)

    loss_fn = LeJEPALoss(lam=lam, num_slices=256)
    total, pred, sigreg = loss_fn(proj)

    expected_total = (1 - lam) * pred.item() + lam * sigreg.item()
    check(
        "total loss matches (1-lambda)*pred + lambda*sigreg",
        math.isclose(total.item(), expected_total, rel_tol=1e-5),
        f"total={total.item():.6f} expected={expected_total:.6f}",
    )

    total.backward()
    check("gradient flows back to proj", proj.grad is not None and proj.grad.abs().sum().item() > 0)


if __name__ == "__main__":
    test_sigreg_distinguishes_gaussian_from_non_gaussian()
    test_sigreg_resamples_directions_every_call()
    test_sigreg_gradient_pulls_toward_gaussian()
    test_jepa_prediction_loss()
    test_lejepa_loss_combines_correctly()

    print()
    if _all_ok:
        print("All checks passed.")
    else:
        print("Some checks FAILED -- see above.")
        sys.exit(1)
