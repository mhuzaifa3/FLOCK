"""Single-frame presentation-attack detection.

Cues are kept separate so the evaluation can report which one carries the
decision rather than reporting one opaque score.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class TextureFeatures:
    laplacian_variance: float
    high_freq_ratio: float
    saturation_std: float
    specular_ratio: float

    def as_vector(self) -> np.ndarray:
        return np.array(
            [
                self.laplacian_variance,
                self.high_freq_ratio,
                self.saturation_std,
                self.specular_ratio,
            ],
            dtype=np.float32,
        )

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _laplacian_variance(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _high_freq_ratio(gray: np.ndarray) -> float:
    """Moire can raise this and blur can lower it, so the value is reported unsigned."""
    f = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
    magnitude = np.abs(f)
    total = float(magnitude.sum())
    if total == 0.0:
        return 0.0

    h, w = gray.shape[:2]
    cy, cx = h // 2, w // 2
    radius = min(h, w) // 8
    mask = np.ones_like(magnitude, dtype=bool)
    cv2.circle(mask.view(np.uint8), (cx, cy), radius, 0, thickness=-1)
    return float(magnitude[mask].sum() / total)


def _saturation_std(bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 1].std())


def _specular_ratio(bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    bright = (hsv[:, :, 2] > 240) & (hsv[:, :, 1] < 40)
    return float(bright.mean())


def extract_features(crop: np.ndarray) -> TextureFeatures:
    if crop is None or crop.size == 0:
        msg = "empty crop"
        raise ValueError(msg)
    if crop.ndim == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return TextureFeatures(
        laplacian_variance=_laplacian_variance(gray),
        high_freq_ratio=_high_freq_ratio(gray),
        saturation_std=_saturation_std(crop),
        specular_ratio=_specular_ratio(crop),
    )
