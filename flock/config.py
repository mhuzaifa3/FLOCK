"""Thresholds and paths. Every value that changes an access decision lives here."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

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
    "face_recognition_sface/face_recognition_sface_2021dec.onnx"
)


@dataclass(frozen=True)
class Thresholds:
    # 0.3388 gives TAR 0.9817 at TRR 0.9980 on the LFW test pairs.
    # Reproduce with: python eval/recognition.py --subset test
    match_cosine: float = 0.3388
    detect_confidence: float = 0.9
    texture_min: float = 0.55
    blink_min_count: int = 1
    blink_window_frames: int = 45
    unlock_seconds: float = 5.0


DEFAULT_THRESHOLDS = Thresholds()
