"""SFace embeddings. Enrollment stores 128-d vectors, never face imagery."""
from __future__ import annotations

import cv2
import numpy as np

from flock.config import EMBEDDER_MODEL
from flock.detect import Face
from flock.modelstore import require_models


class FaceEmbedder:
    def __init__(self) -> None:
        require_models()
        self._recognizer = cv2.FaceRecognizerSF_create(str(EMBEDDER_MODEL), "")

    def embed(self, frame: np.ndarray, face: Face) -> np.ndarray:
        aligned = self._recognizer.alignCrop(frame, face.raw)
        feature = self._recognizer.feature(aligned)
        return np.asarray(feature, dtype=np.float32).reshape(-1)

    def aligned_crop(self, frame: np.ndarray, face: Face) -> np.ndarray:
        return self._recognizer.alignCrop(frame, face.raw)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Computed directly so stored templates compare without a model handle."""
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)
