"""Model weights are downloaded on first use rather than committed."""
from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

from flock.config import DETECTOR_MODEL, DETECTOR_URL, EMBEDDER_MODEL, EMBEDDER_URL

logger = logging.getLogger(__name__)

_MODELS = ((DETECTOR_MODEL, DETECTOR_URL), (EMBEDDER_MODEL, EMBEDDER_URL))


def ensure_models(force: bool = False) -> list[Path]:
    paths = []
    for path, url in _MODELS:
        if force or not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("fetching %s", path.name)
            urllib.request.urlretrieve(url, path)
        paths.append(path)
    return paths


def require_models() -> None:
    missing = [p.name for p, _ in _MODELS if not p.exists()]
    if missing:
        msg = (
            f"Missing model weights: {', '.join(missing)}. "
            "Run `python scripts/fetch_models.py` first."
        )
        raise FileNotFoundError(msg)
