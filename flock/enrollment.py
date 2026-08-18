"""Enrolled identities, stored as embeddings rather than images."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from flock.config import ENROLLMENT_DIR
from flock.embed import cosine_similarity


@dataclass(frozen=True)
class Match:
    name: str
    similarity: float
    accepted: bool


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

    def identify(self, embedding: np.ndarray, threshold: float) -> Match:
        best_name, best_score = "", -1.0
        for name, template in self.templates().items():
            score = cosine_similarity(embedding, template)
            if score > best_score:
                best_name, best_score = name, score
        if not best_name:
            return Match(name="", similarity=0.0, accepted=False)
        return Match(name=best_name, similarity=best_score, accepted=best_score >= threshold)

    def export_manifest(self) -> str:
        return json.dumps({"enrolled": self.names()}, indent=2)
