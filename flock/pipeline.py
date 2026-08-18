"""Access decision pipeline.

Liveness is evaluated before identity, so a spoof of an enrolled face is
rejected without the identity ever being considered.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flock.config import DEFAULT_THRESHOLDS, Thresholds
from flock.detect import Face, FaceDetector
from flock.embed import FaceEmbedder
from flock.enrollment import EnrollmentStore
from flock.events import AccessEvent, EventSink
from flock.liveness.blink import BlinkDetector, eye_openness
from flock.liveness.texture import TextureFeatures, extract_features
from flock.lock import Lock

DENY_NO_FACE = "no_face"
DENY_TEXTURE = "texture_spoof"
DENY_NO_BLINK = "no_blink"
DENY_UNKNOWN = "unknown_identity"
ALLOW = "match"


@dataclass(frozen=True)
class Decision:
    unlocked: bool
    reason: str
    identity: str = ""
    similarity: float = 0.0
    texture_score: float = 0.0
    blink_count: int = 0

    def to_event(self) -> AccessEvent:
        return AccessEvent(
            decision="unlock" if self.unlocked else "deny",
            reason=self.reason,
            identity=self.identity,
            similarity=round(self.similarity, 4),
            texture_score=round(self.texture_score, 4),
            blink_count=self.blink_count,
        )


def texture_score(features: TextureFeatures) -> float:
    """Maps raw cues to [0, 1]. Calibrate with eval/liveness.py on your camera."""
    sharpness = min(features.laplacian_variance / 300.0, 1.0)
    detail = min(features.high_freq_ratio / 0.75, 1.0)
    colour = min(features.saturation_std / 45.0, 1.0)
    glare = max(0.0, 1.0 - features.specular_ratio * 12.0)
    return float(np.average([sharpness, detail, colour, glare], weights=[0.4, 0.3, 0.2, 0.1]))


class AccessPipeline:
    def __init__(
        self,
        detector: FaceDetector,
        embedder: FaceEmbedder,
        store: EnrollmentStore,
        lock: Lock,
        sink: EventSink | None = None,
        thresholds: Thresholds = DEFAULT_THRESHOLDS,
    ) -> None:
        self.detector = detector
        self.embedder = embedder
        self.store = store
        self.lock = lock
        self.sink = sink
        self.thresholds = thresholds
        self.blink = BlinkDetector(window_frames=thresholds.blink_window_frames)

    def process(self, frame: np.ndarray) -> Decision:
        face = self.detector.detect_primary(frame)
        if face is None:
            self.blink.reset()
            return self._finish(Decision(unlocked=False, reason=DENY_NO_FACE))

        self.blink.update(eye_openness(frame, face))
        blinks = self.blink.blink_count()

        crop = self.embedder.aligned_crop(frame, face)
        score = texture_score(extract_features(crop))
        if score < self.thresholds.texture_min:
            return self._finish(
                Decision(False, DENY_TEXTURE, texture_score=score, blink_count=blinks),
            )
        if blinks < self.thresholds.blink_min_count:
            return self._finish(
                Decision(False, DENY_NO_BLINK, texture_score=score, blink_count=blinks),
            )

        match = self.store.identify(self.embedder.embed(frame, face), self.thresholds.match_cosine)
        if not match.accepted:
            return self._finish(
                Decision(False, DENY_UNKNOWN, similarity=match.similarity,
                         texture_score=score, blink_count=blinks),
            )

        self.lock.unlock(self.thresholds.unlock_seconds)
        self.blink.reset()
        return self._finish(
            Decision(True, ALLOW, identity=match.name, similarity=match.similarity,
                     texture_score=score, blink_count=blinks),
        )

    def _finish(self, decision: Decision) -> Decision:
        if self.sink is not None:
            self.sink.emit(decision.to_event())
        return decision

    def last_face(self, frame: np.ndarray) -> Face | None:
        return self.detector.detect_primary(frame)
