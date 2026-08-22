"""Enrolled identities, stored as embeddings rather than images.

Search is one-to-many, so every enrolled template is another chance for a
stranger to match. The threshold is drawn against the roster size for that
reason, and a probe that fits two people well enough to confuse them is
refused rather than resolved to whichever scored higher.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from flock.config import DEFAULT_THRESHOLDS, ENROLLMENT_DIR, Thresholds
from flock.embed import cosine_similarity

AMBIGUOUS = "ambiguous"
NO_ENROLLMENTS = "no_enrollments"
BELOW_THRESHOLD = "below_threshold"
ACCEPTED = "accepted"


@dataclass(frozen=True)
class Match:
    name: str
    similarity: float
    accepted: bool
    outcome: str = BELOW_THRESHOLD
    threshold: float = 0.0
    runner_up: str = ""
    runner_up_similarity: float = 0.0
    roster_size: int = 0

    @property
    def margin(self) -> float:
        return self.similarity - self.runner_up_similarity


class EnrollmentStore:
    def __init__(self, directory: Path | None = None) -> None:
        self.directory = Path(directory or ENROLLMENT_DIR)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        safe = "".join(c for c in name if c.isalnum() or c in "-_")
        if not safe:
            msg = f"unusable enrollment name: {name!r}"
            raise ValueError(msg)
        return self.directory / f"{safe}.npz"

    def enroll(self, name: str, embeddings: list[np.ndarray]) -> Path:
        """Stores the mean of several samples, which is more stable than one."""
        if not embeddings:
            msg = "no embeddings supplied"
            raise ValueError(msg)
        stacked = np.vstack([np.asarray(e, dtype=np.float32).reshape(1, -1) for e in embeddings])
        template = stacked.mean(axis=0)
        template /= np.linalg.norm(template) or 1.0
        path = self._path(name)
        np.savez(path, template=template, samples=len(embeddings))
        return path

    def remove(self, name: str) -> bool:
        path = self._path(name)
        if path.exists():
            path.unlink()
            return True
        return False

    def names(self) -> list[str]:
        return sorted(p.stem for p in self.directory.glob("*.npz"))

    def templates(self) -> dict[str, np.ndarray]:
        loaded = {}
        for path in self.directory.glob("*.npz"):
            with np.load(path) as data:
                loaded[path.stem] = data["template"]
        return loaded

    def identify(self, embedding: np.ndarray, thresholds: Thresholds = DEFAULT_THRESHOLDS) -> Match:
        ranked = sorted(
            ((cosine_similarity(embedding, t), n) for n, t in self.templates().items()),
            reverse=True,
        )
        if not ranked:
            return Match(name="", similarity=0.0, accepted=False, outcome=NO_ENROLLMENTS)

        threshold = thresholds.match_cosine_for_roster(len(ranked))
        best_score, best_name = ranked[0]
        second_score, second_name = ranked[1] if len(ranked) > 1 else (0.0, "")
        partial = Match(
            name=best_name,
            similarity=best_score,
            accepted=False,
            threshold=threshold,
            runner_up=second_name,
            runner_up_similarity=second_score,
            roster_size=len(ranked),
        )

        if best_score < threshold:
            return partial
        if second_name and best_score - second_score < thresholds.identify_margin:
            return replace(partial, outcome=AMBIGUOUS)
        return replace(partial, accepted=True, outcome=ACCEPTED)

    def export_manifest(self) -> str:
        return json.dumps({"enrolled": self.names()}, indent=2)
