"""Thresholds and paths. Every value that changes an access decision lives here."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from flock.calibrate import is_extrapolated, threshold_for_fmr

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.environ.get("FLOCK_MODEL_DIR", REPO_ROOT / "models"))
ENROLLMENT_DIR = Path(os.environ.get("FLOCK_ENROLLMENT_DIR", REPO_ROOT / "enrollments"))

DETECTOR_MODEL = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
EMBEDDER_MODEL = MODEL_DIR / "face_recognition_sface_2021dec.onnx"

DETECTOR_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
EMBEDDER_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_recognition_sface_2021dec.onnx"
)


@dataclass(frozen=True)
class Thresholds:
    target_false_match_rate: float = 1e-6
    match_cosine_override: float | None = None
    detect_confidence: float = 0.9
    texture_min: float = 0.55
    blink_min_count: int = 1
    blink_window_frames: int = 45
    unlock_seconds: float = 5.0

    @property
    def match_cosine(self) -> float:
        if self.match_cosine_override is not None:
            return self.match_cosine_override
        return threshold_for_fmr(self.target_false_match_rate)

    @property
    def match_is_extrapolated(self) -> bool:
        if self.match_cosine_override is not None:
            return False
        return is_extrapolated(self.target_false_match_rate)


DEFAULT_THRESHOLDS = Thresholds()
