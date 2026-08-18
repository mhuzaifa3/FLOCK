import numpy as np
import pytest

from flock.config import Thresholds
from flock.enrollment import EnrollmentStore
from flock.lock import SimulatedLock
from flock.pipeline import (
    ALLOW,
    DENY_NO_BLINK,
    DENY_NO_FACE,
    DENY_TEXTURE,
    DENY_UNKNOWN,
    AccessPipeline,
)
from tests.conftest import FakeDetector, FakeEmbedder, RecordingSink, make_face

ENROLLED = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
STRANGER = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)


@pytest.fixture
def store(tmp_path):
    enrollment = EnrollmentStore(tmp_path / "enrollments")
    enrollment.enroll("alice", [ENROLLED])
    return enrollment


def build(store, embedding, crop, face=None, thresholds=None):
    lock, sink = SimulatedLock(), RecordingSink()
    pipeline = AccessPipeline(
        detector=FakeDetector(make_face() if face is None else face),
        embedder=FakeEmbedder(embedding, crop),
        store=store,
        lock=lock,
        sink=sink,
        thresholds=thresholds or Thresholds(blink_min_count=0),
    )
    return pipeline, lock, sink


def test_no_face_denies(store, frame):
    pipeline, lock, sink = build(store, ENROLLED, "sharp", face=False)
    pipeline.detector = FakeDetector(None)
    decision = pipeline.process(frame)
    assert not decision.unlocked
    assert decision.reason == DENY_NO_FACE
    assert lock.unlock_calls == []
    assert sink.events[-1].decision == "deny"


def test_flat_crop_is_rejected_as_spoof(store, frame):
    pipeline, lock, _ = build(store, ENROLLED, crop=200)
    decision = pipeline.process(frame)
    assert not decision.unlocked
    assert decision.reason == DENY_TEXTURE
    assert lock.unlock_calls == []


def test_enrolled_user_unlocks(store, frame):
    pipeline, lock, sink = build(store, ENROLLED, crop="sharp")
    decision = pipeline.process(frame)
    assert decision.unlocked
    assert decision.reason == ALLOW
    assert decision.identity == "alice"
    assert lock.unlock_calls == [Thresholds().unlock_seconds]
    assert sink.events[-1].decision == "unlock"


def test_stranger_is_denied(store, frame):
    pipeline, lock, _ = build(store, STRANGER, crop="sharp")
    decision = pipeline.process(frame)
    assert not decision.unlocked
    assert decision.reason == DENY_UNKNOWN
    assert lock.unlock_calls == []


def test_liveness_is_checked_before_identity(store, frame):
    """A spoof of an enrolled face must fail on liveness, not reach identity."""
    pipeline, lock, _ = build(store, ENROLLED, crop=200)
    decision = pipeline.process(frame)
    assert decision.reason == DENY_TEXTURE
    assert decision.identity == ""
    assert lock.unlock_calls == []


def test_blink_requirement_denies_still_image(store, frame):
    pipeline, lock, _ = build(
        store, ENROLLED, crop="sharp", thresholds=Thresholds(blink_min_count=1),
    )
    decision = pipeline.process(frame)
    assert not decision.unlocked
    assert decision.reason == DENY_NO_BLINK
    assert lock.unlock_calls == []


def test_events_never_carry_biometric_data(store, frame):
    pipeline, _, sink = build(store, ENROLLED, crop="sharp")
    pipeline.process(frame)
    payload = sink.events[-1].as_json()
    assert "embedding" not in payload
    assert "template" not in payload
    assert "image" not in payload
