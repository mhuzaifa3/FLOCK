"""Blink-based liveness.

YuNet gives one landmark per eye, not the six-point contour eye-aspect-ratio
needs, so openness is measured from the eye patch: an open eye carries the
iris/sclera edge, a closed lid replaces it with flat skin.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from flock.detect import Face


def _eye_patch(frame: np.ndarray, centre: tuple[float, float], half: int) -> np.ndarray:
    h, w = frame.shape[:2]
    cx, cy = round(centre[0]), round(centre[1])
    x0, x1 = max(0, cx - half), min(w, cx + half)
    y0, y1 = max(0, cy - half), min(h, cy + half)
    if x1 <= x0 or y1 <= y0:
        return np.empty((0, 0), dtype=np.uint8)
    patch = frame[y0:y1, x0:x1]
    if patch.ndim == 3:
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return patch


def _patch_openness(patch: np.ndarray) -> float:
    """Normalised by mean intensity so the baseline tracks the subject, not the room."""
    if patch.size == 0:
        return 0.0
    patch = patch.astype(np.float32)
    mean = float(patch.mean())
    if mean <= 1e-6:
        return 0.0
    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.sqrt(gx**2 + gy**2).mean() / mean)


def eye_openness(frame: np.ndarray, face: Face) -> float:
    dx = face.left_eye[0] - face.right_eye[0]
    dy = face.left_eye[1] - face.right_eye[1]
    interocular = float(np.hypot(dx, dy))
    half = max(4, round(interocular * 0.28))
    values = [
        _patch_openness(_eye_patch(frame, face.right_eye, half)),
        _patch_openness(_eye_patch(frame, face.left_eye, half)),
    ]
    return float(np.mean(values))


@dataclass
class BlinkState:
    baseline: float
    is_closed: bool
    blink_count: int


class BlinkDetector:
    """Blinks are transient dips below a rolling baseline.

    The baseline is a median over samples classified as open, so a long closure
    cannot drag the reference down with it.
    """

    def __init__(
        self,
        window_frames: int = 45,
        dip_ratio: float = 0.72,
        min_closed_frames: int = 1,
        max_closed_frames: int = 12,
        warmup_frames: int = 8,
    ) -> None:
        self.window_frames = window_frames
        self.dip_ratio = dip_ratio
        self.min_closed_frames = min_closed_frames
        self.max_closed_frames = max_closed_frames
        self.warmup_frames = warmup_frames
        self._open_samples: deque[float] = deque(maxlen=window_frames)
        self._blinks: deque[int] = deque(maxlen=window_frames)
        self._frame_index = 0
        self._closed_run = 0

    @property
    def baseline(self) -> float:
        if not self._open_samples:
            return 0.0
        return float(np.median(self._open_samples))

    def blink_count(self) -> int:
        cutoff = self._frame_index - self.window_frames
        return sum(1 for idx in self._blinks if idx > cutoff)

    def reset(self) -> None:
        self._open_samples.clear()
        self._blinks.clear()
        self._frame_index = 0
        self._closed_run = 0

    def update(self, openness: float) -> bool:
        """Returns True on the frame a blink completes."""
        self._frame_index += 1
        blinked = False

        if len(self._open_samples) < self.warmup_frames:
            self._open_samples.append(openness)
            return False

        threshold = self.baseline * self.dip_ratio
        if openness < threshold:
            self._closed_run += 1
        else:
            if self.min_closed_frames <= self._closed_run <= self.max_closed_frames:
                self._blinks.append(self._frame_index)
                blinked = True
            self._closed_run = 0
            self._open_samples.append(openness)

        return blinked

    def state(self) -> BlinkState:
        return BlinkState(
            baseline=self.baseline,
            is_closed=self._closed_run > 0,
            blink_count=self.blink_count(),
        )
