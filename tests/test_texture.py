import cv2
import numpy as np
import pytest

from flock.liveness.texture import extract_features
from flock.pipeline import texture_score


def _sharp():
    return np.random.default_rng(0).integers(0, 255, (112, 112, 3), dtype=np.uint8)


def test_blur_reduces_detail_cues():
    sharp = extract_features(_sharp())
    blurred = extract_features(cv2.GaussianBlur(_sharp(), (9, 9), 0))
    assert blurred.laplacian_variance < sharp.laplacian_variance
    assert blurred.high_freq_ratio < sharp.high_freq_ratio


def test_flat_image_scores_below_sharp_image():
    flat = np.full((112, 112, 3), 200, dtype=np.uint8)
    assert texture_score(extract_features(flat)) < texture_score(extract_features(_sharp()))


def test_grayscale_input_is_accepted():
    gray = cv2.cvtColor(_sharp(), cv2.COLOR_BGR2GRAY)
    assert extract_features(gray).laplacian_variance > 0


def test_empty_crop_raises():
    with pytest.raises(ValueError, match="empty crop"):
        extract_features(np.empty((0, 0, 3), dtype=np.uint8))


def test_score_is_bounded():
    for image in (_sharp(), np.zeros((112, 112, 3), dtype=np.uint8)):
        assert 0.0 <= texture_score(extract_features(image)) <= 1.0
