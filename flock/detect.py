"""Face detection with YuNet."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from flock.config import DEFAULT_THRESHOLDS, DETECTOR_MODEL, Thresholds
from flock.modelstore import require_models


@dataclass(frozen=True)
class Face:
    bbox: tuple[int, int, int, int]
    right_eye: tuple[float, float]
    left_eye: tuple[float, float]
    nose: tuple[float, float]
    mouth_right: tuple[float, float]
    mouth_left: tuple[float, float]
    confidence: float
    raw: np.ndarray

    @property
    def area(self) -> int:
        return self.bbox[2] * self.bbox[3]


class FaceDetector:
    def __init__(self, thresholds: Thresholds = DEFAULT_THRESHOLDS) -> None:
        require_models()
        self._thresholds = thresholds
        self._detector = cv2.FaceDetectorYN_create(
            str(DETECTOR_MODEL),
            "",
            (320, 320),
            score_threshold=thresholds.detect_confidence,
        )
        self._size: tuple[int, int] | None = None

    def detect(self, frame: np.ndarray) -> list[Face]:
        if frame is None or frame.size == 0:
            return []
        height, width = frame.shape[:2]
        if self._size != (width, height):
            self._detector.setInputSize((width, height))
            self._size = (width, height)

        _, raw = self._detector.detect(frame)
        if raw is None:
            return []

        faces = [self._to_face(row) for row in raw]
        return sorted(faces, key=lambda f: f.area, reverse=True)

    def detect_primary(self, frame: np.ndarray) -> Face | None:
        """Largest face, which at a door is the person standing at it."""
        faces = self.detect(frame)
        return faces[0] if faces else None

    @staticmethod
    def _to_face(row: np.ndarray) -> Face:
        x, y, w, h = (int(v) for v in row[:4])
        return Face(
            bbox=(x, y, w, h),
            right_eye=(float(row[4]), float(row[5])),
            left_eye=(float(row[6]), float(row[7])),
            nose=(float(row[8]), float(row[9])),
            mouth_right=(float(row[10]), float(row[11])),
            mouth_left=(float(row[12]), float(row[13])),
            confidence=float(row[14]),
            raw=row.copy(),
        )
