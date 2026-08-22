import numpy as np
import pytest

from flock.calibrate import threshold_for_fmr
from flock.config import Thresholds
from flock.enrollment import ACCEPTED, AMBIGUOUS, NO_ENROLLMENTS, EnrollmentStore

A = np.array([1.0, 0.0, 0.0], dtype=np.float32)
B = np.array([0.0, 1.0, 0.0], dtype=np.float32)

LOOSE = Thresholds(match_cosine_override=0.3)


def basis(index: int, size: int = 24) -> np.ndarray:
    vector = np.zeros(size, dtype=np.float32)
    vector[index] = 1.0
    return vector


def probe_at(cosine: float, size: int = 24) -> np.ndarray:
    vector = np.zeros(size, dtype=np.float32)
    vector[0] = cosine
    vector[-1] = float(np.sqrt(1.0 - cosine**2))
    return vector


@pytest.fixture
def store(tmp_path):
    return EnrollmentStore(tmp_path)


def test_enroll_and_identify(store):
    store.enroll("alice", [A, A])
    match = store.identify(A, LOOSE)
    assert match.name == "alice"
    assert match.accepted
    assert match.outcome == ACCEPTED


def test_unknown_embedding_is_rejected(store):
    store.enroll("alice", [A])
    assert not store.identify(B, LOOSE).accepted


def test_identify_with_empty_store(store):
    match = store.identify(A, LOOSE)
    assert match.name == ""
    assert not match.accepted
    assert match.outcome == NO_ENROLLMENTS


def test_template_averages_samples(store):
    store.enroll("alice", [A, B])
    template = store.templates()["alice"]
    assert pytest.approx(float(np.linalg.norm(template)), abs=1e-5) == 1.0


def test_path_traversal_cannot_escape_the_store(store, tmp_path):
    store.enroll("../../etc/passwd", [A])
    written = list(tmp_path.rglob("*.npz"))
    assert len(written) == 1
    assert written[0].parent == tmp_path
    assert store.names() == ["etcpasswd"]


def test_rejects_unusable_name(store):
    with pytest.raises(ValueError, match="unusable"):
        store.enroll("///", [A])


def test_rejects_empty_embeddings(store):
    with pytest.raises(ValueError, match="no embeddings"):
        store.enroll("alice", [])


def test_remove(store):
    store.enroll("alice", [A])
    assert store.remove("alice")
    assert not store.remove("alice")
    assert store.names() == []


def test_threshold_rises_with_the_roster():
    solo = Thresholds().match_cosine_for_roster(1)
    twenty = Thresholds().match_cosine_for_roster(20)
    assert twenty > solo
    assert twenty == threshold_for_fmr(1e-6 / 20)


def test_a_match_that_passes_alone_fails_against_a_full_roster(tmp_path):
    """The LFW threshold is one-to-one; twenty templates is twenty chances."""
    probe = probe_at(0.56)
    solo = EnrollmentStore(tmp_path / "solo")
    solo.enroll("alice", [basis(0)])
    assert solo.identify(probe).accepted

    crowd = EnrollmentStore(tmp_path / "crowd")
    for index in range(20):
        crowd.enroll(f"person{index}", [basis(index)])
    outcome = crowd.identify(probe)
    assert not outcome.accepted
    assert outcome.name == "person0"
    assert outcome.roster_size == 20
    assert outcome.threshold > solo.identify(probe).threshold


def test_two_people_the_probe_cannot_separate_are_refused(tmp_path):
    store = EnrollmentStore(tmp_path)
    store.enroll("alice", [basis(0)])
    store.enroll("alicia", [basis(0)])
    outcome = store.identify(basis(0))
    assert not outcome.accepted
    assert outcome.outcome == AMBIGUOUS
    assert outcome.margin == pytest.approx(0.0)


def test_a_clear_winner_over_a_full_roster_is_accepted(tmp_path):
    store = EnrollmentStore(tmp_path)
    for index in range(20):
        store.enroll(f"person{index}", [basis(index)])
    outcome = store.identify(basis(7))
    assert outcome.accepted
    assert outcome.name == "person7"
    assert outcome.margin == pytest.approx(1.0)


def test_match_reports_the_runner_up(tmp_path):
    store = EnrollmentStore(tmp_path)
    store.enroll("alice", [basis(0)])
    store.enroll("bob", [probe_at(0.5)])
    outcome = store.identify(basis(0))
    assert outcome.name == "alice"
    assert outcome.runner_up == "bob"
    assert outcome.runner_up_similarity == pytest.approx(0.5, abs=1e-6)
