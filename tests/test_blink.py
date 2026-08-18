from flock.liveness.blink import BlinkDetector


def _run(sequence):
    detector = BlinkDetector()
    return sum(detector.update(value) for value in sequence), detector


def test_blinks_are_counted():
    count, detector = _run([1.0] * 10 + [0.4, 0.35] + [1.0] * 10 + [0.3] + [1.0] * 10)
    assert count == 2
    assert detector.blink_count() == 2


def test_photograph_never_blinks():
    count, _ = _run([1.0] * 60)
    assert count == 0


def test_long_occlusion_is_not_a_blink():
    count, _ = _run([1.0] * 10 + [0.05] * 40)
    assert count == 0


def test_reset_clears_history():
    _, detector = _run([1.0] * 10 + [0.3] + [1.0] * 5)
    assert detector.blink_count() == 1
    detector.reset()
    assert detector.blink_count() == 0
    assert detector.baseline == 0.0


def test_noise_around_baseline_is_not_a_blink():
    count, _ = _run([1.0, 0.98, 1.02, 0.97, 1.01] * 12)
    assert count == 0
