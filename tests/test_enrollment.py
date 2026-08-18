import numpy as np
import pytest

from flock.enrollment import EnrollmentStore

A = np.array([1.0, 0.0, 0.0], dtype=np.float32)
B = np.array([0.0, 1.0, 0.0], dtype=np.float32)


@pytest.fixture
def store(tmp_path):
    return EnrollmentStore(tmp_path)


def test_enroll_and_identify(store):
    store.enroll("alice", [A, A])
    match = store.identify(A, threshold=0.3)
    assert match.name == "alice"
    assert match.accepted


def test_unknown_embedding_is_rejected(store):
    store.enroll("alice", [A])
    assert not store.identify(B, threshold=0.3).accepted


def test_identify_with_empty_store(store):
    match = store.identify(A, threshold=0.3)
    assert match.name == ""
    assert not match.accepted


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
