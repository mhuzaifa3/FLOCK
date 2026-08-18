"""Fakes for the vision stages, so pipeline logic is testable without face data."""
from __future__ import annotations

import numpy as np
import pytest

from flock.detect import Face


def make_face(bbox=(10, 10, 60, 60), confidence=0.99) -> Face:
    raw = np.zeros(15, dtype=np.float32)
    raw[:4] = bbox
    raw[4:6] = (bbox[0] + 15, bbox[1] + 20)
    raw[6:8] = (bbox[0] + 45, bbox[1] + 20)
    raw[14] = confidence
    return Face(
        bbox=bbox,
        right_eye=(bbox[0] + 15, bbox[1] + 20),
        left_eye=(bbox[0] + 45, bbox[1] + 20),
        nose=(bbox[0] + 30, bbox[1] + 35),
        mouth_right=(bbox[0] + 20, bbox[1] + 50),
        mouth_left=(bbox[0] + 40, bbox[1] + 50),
        confidence=confidence,
        raw=raw,
    )


class FakeDetector:
    def __init__(self, face: Face | None) -> None:
        self._face = face

    def detect_primary(self, frame):
        return self._face


class FakeEmbedder:
    """Returns a fixed embedding and a crop whose texture score is controllable."""

    def __init__(self, embedding: np.ndarray, crop_value: int = 200) -> None:
        self._embedding = embedding
        self._crop_value = crop_value

    def embed(self, frame, face):
        return self._embedding

    def aligned_crop(self, frame, face):
        rng = np.random.default_rng(0)
        if self._crop_value == "sharp":
            return rng.integers(0, 255, (112, 112, 3), dtype=np.uint8)
        return np.full((112, 112, 3), self._crop_value, dtype=np.uint8)


class RecordingSink:
    def __init__(self) -> None:
        self.events = []

    def emit(self, event) -> None:
        self.events.append(event)


@pytest.fixture
def frame() -> np.ndarray:
    return np.zeros((240, 320, 3), dtype=np.uint8)
