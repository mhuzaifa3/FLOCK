"""Turns a target false-match rate into a cosine threshold.

A door lock is specified by how often a stranger gets in, so that is the number
the configuration carries. The mapping from that rate to a cosine threshold
comes from the impostor score distribution measured on the LFW test pairs.

The measurement has a floor. 495 impostor comparisons cannot resolve a rate
finer than about 1 in 500, so any rate below that is an extrapolation of the
fitted tail, not an observation. ``fmr_upper_bound`` reports what the data
actually supports; treat the gap between the two as unproven.
"""
from __future__ import annotations

from math import exp, lgamma, log
from statistics import NormalDist

IMPOSTOR_MEAN = 0.084015
IMPOSTOR_STDDEV = 0.095297
IMPOSTOR_COMPARISONS = 495

MEASURED_FMR_FLOOR = 0.0060

_NORMAL = NormalDist()


def threshold_for_fmr(fmr: float) -> float:
    """Cosine threshold whose modelled per-comparison false-match rate is ``fmr``."""
    if not 0.0 < fmr < 1.0:
        msg = f"false-match rate must be in (0, 1), got {fmr!r}"
        raise ValueError(msg)
    return IMPOSTOR_MEAN + IMPOSTOR_STDDEV * _NORMAL.inv_cdf(1.0 - fmr)


def fmr_for_threshold(threshold: float) -> float:
    return float(_NORMAL.cdf(-(threshold - IMPOSTOR_MEAN) / IMPOSTOR_STDDEV))


def is_extrapolated(fmr: float) -> bool:
    return fmr < MEASURED_FMR_FLOOR


def _binomial_cdf(successes: int, trials: int, p: float) -> float:
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0 if successes < trials else 1.0
    total = 0.0
    for k in range(successes + 1):
        log_term = (
            lgamma(trials + 1)
            - lgamma(k + 1)
            - lgamma(trials - k + 1)
            + k * log(p)
            + (trials - k) * log(1.0 - p)
        )
        total += exp(log_term)
    return min(total, 1.0)


def fmr_upper_bound(accepts: int, comparisons: int, confidence: float = 0.95) -> float:
    """Clopper-Pearson upper bound on the true false-match rate.

    With zero accepts this is the rule of three: the best a run of ``n``
    comparisons can claim is roughly 3/n, however clean the result looks.
    """
    if comparisons <= 0:
        msg = "need at least one comparison"
        raise ValueError(msg)
    if not 0 <= accepts <= comparisons:
        msg = f"accepts {accepts} outside 0..{comparisons}"
        raise ValueError(msg)
    if accepts == comparisons:
        return 1.0

    alpha = 1.0 - confidence
    low, high = 0.0, 1.0
    for _ in range(200):
        mid = (low + high) / 2.0
        if _binomial_cdf(accepts, comparisons, mid) > alpha:
            low = mid
        else:
            high = mid
    return high


def comparisons_for_fmr(fmr: float, confidence: float = 0.95) -> int:
    """Impostor comparisons needed to demonstrate ``fmr`` with zero accepts."""
    if not 0.0 < fmr < 1.0:
        msg = f"false-match rate must be in (0, 1), got {fmr!r}"
        raise ValueError(msg)
    return int(-log(1.0 - confidence) / -log(1.0 - fmr)) + 1
