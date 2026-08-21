import pytest

from flock.calibrate import (
    MEASURED_FMR_FLOOR,
    comparisons_for_fmr,
    fmr_for_threshold,
    fmr_upper_bound,
    is_extrapolated,
    threshold_for_fmr,
)
from flock.config import DEFAULT_THRESHOLDS, Thresholds


def test_stricter_rate_demands_a_higher_threshold():
    thresholds = [threshold_for_fmr(r) for r in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)]
    assert thresholds == sorted(thresholds)


def test_threshold_and_rate_round_trip():
    for rate in (1e-2, 1e-4, 1e-6):
        assert fmr_for_threshold(threshold_for_fmr(rate)) == pytest.approx(rate, rel=1e-6)


def test_rate_below_the_measurement_floor_is_flagged():
    assert is_extrapolated(1e-6)
    assert not is_extrapolated(1e-2)


@pytest.mark.parametrize("rate", [0.0, 1.0, -0.1, 1.5])
def test_impossible_rates_are_rejected(rate):
    with pytest.raises(ValueError, match="false-match rate"):
        threshold_for_fmr(rate)


def test_zero_accepts_still_bounds_the_rate_at_three_over_n():
    """A clean run of 495 comparisons proves nothing below roughly 1 in 165."""
    assert fmr_upper_bound(0, 495) == pytest.approx(3.0 / 495, rel=0.05)


def test_one_accept_in_494_bounds_near_one_percent():
    assert fmr_upper_bound(1, 494) == pytest.approx(0.0096, abs=5e-4)


def test_more_comparisons_tighten_the_bound():
    assert fmr_upper_bound(0, 10_000) < fmr_upper_bound(0, 495)


def test_bound_is_a_bound():
    assert fmr_upper_bound(5, 1000) > 5 / 1000


def test_saturated_bound():
    assert fmr_upper_bound(10, 10) == 1.0


@pytest.mark.parametrize(("accepts", "comparisons"), [(1, 0), (-1, 10), (11, 10)])
def test_nonsense_counts_are_rejected(accepts, comparisons):
    with pytest.raises(ValueError):
        fmr_upper_bound(accepts, comparisons)


def test_demonstrating_one_in_a_million_needs_millions_of_comparisons():
    assert comparisons_for_fmr(1e-6) > 2_000_000


def test_measurement_floor_matches_the_calibration_run():
    assert MEASURED_FMR_FLOOR == pytest.approx(fmr_upper_bound(0, 495), abs=5e-5)


def test_shipped_threshold_comes_from_the_target_rate():
    assert DEFAULT_THRESHOLDS.match_cosine == threshold_for_fmr(1e-6)
    assert DEFAULT_THRESHOLDS.match_is_extrapolated


def test_shipped_threshold_clears_every_observed_lfw_impostor():
    """The old default sat exactly on the highest impostor score in the run."""
    assert DEFAULT_THRESHOLDS.match_cosine > 0.3388


def test_override_pins_the_threshold():
    pinned = Thresholds(match_cosine_override=0.42)
    assert pinned.match_cosine == 0.42
    assert not pinned.match_is_extrapolated
