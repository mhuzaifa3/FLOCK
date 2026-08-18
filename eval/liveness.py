"""Measures presentation-attack detection on a locally captured dataset.

No spoof corpus ships with this repository. Capture your own:

    data/liveness/live/     frames of a real face at the camera
    data/liveness/spoof/    the same face re-presented from a phone or a print

Reports the true-accept rate on live frames and the true-reject rate on spoofs
across a threshold sweep, and prints the per-cue separation so it is visible
which signal is doing the work.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from flock.detect import FaceDetector
from flock.embed import FaceEmbedder
from flock.liveness.texture import extract_features
from flock.pipeline import texture_score

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def scores_for(directory: Path, detector: FaceDetector, embedder: FaceEmbedder):
    scores, features, skipped = [], [], 0
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        frame = cv2.imread(str(path))
        if frame is None:
            skipped += 1
            continue
        face = detector.detect_primary(frame)
        if face is None:
            skipped += 1
            continue
        cue = extract_features(embedder.aligned_crop(frame, face))
        features.append(cue.as_vector())
        scores.append(texture_score(cue))
    return np.array(scores), np.array(features), skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/liveness"))
    args = parser.parse_args()

    live_dir, spoof_dir = args.dataset / "live", args.dataset / "spoof"
    if not live_dir.is_dir() or not spoof_dir.is_dir():
        print(f"expected {live_dir} and {spoof_dir}; see the module docstring")
        return 1

    detector, embedder = FaceDetector(), FaceEmbedder()
    live, live_features, live_skipped = scores_for(live_dir, detector, embedder)
    spoof, spoof_features, spoof_skipped = scores_for(spoof_dir, detector, embedder)

    if live.size == 0 or spoof.size == 0:
        print("need at least one usable image in each class")
        return 1

    print(f"live frames  {live.size} (skipped {live_skipped}), mean score {live.mean():.3f}")
    print(f"spoof frames {spoof.size} (skipped {spoof_skipped}), mean score {spoof.mean():.3f}")
    print()
    names = ["laplacian_variance", "high_freq_ratio", "saturation_std", "specular_ratio"]
    print(f"{'cue':>20} {'live mean':>12} {'spoof mean':>12}")
    for index, name in enumerate(names):
        print(f"{name:>20} {live_features[:, index].mean():>12.4f} "
              f"{spoof_features[:, index].mean():>12.4f}")
    print()
    print(f"{'threshold':>10} {'TAR(live)':>11} {'TRR(spoof)':>12}")
    for threshold in np.arange(0.30, 0.85, 0.05):
        tar = float((live >= threshold).mean())
        trr = float((spoof < threshold).mean())
        print(f"{threshold:>10.2f} {tar:>11.4f} {trr:>12.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
