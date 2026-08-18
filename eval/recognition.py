"""Measures verification accuracy on the LFW pairs benchmark.

Reports the true-accept rate at a fixed true-reject rate, which is the way a
door lock is actually tuned: fix how often a stranger gets in, then see how
often the owner has to try twice.

The dataset is downloaded to the scikit-learn cache, not into this repository.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_lfw_pairs

from flock.detect import FaceDetector
from flock.embed import FaceEmbedder, cosine_similarity


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if image.max() <= 1.0:
        image = image * 255.0
    return image[:, :, ::-1].astype(np.uint8)


def pair_scores(subset: str, limit: int | None) -> tuple[np.ndarray, np.ndarray, int]:
    data = fetch_lfw_pairs(subset=subset, color=True, resize=1.0, slice_=None, funneled=True)
    pairs, labels = data.pairs, data.target
    if limit:
        pairs, labels = pairs[:limit], labels[:limit]

    detector, embedder = FaceDetector(), FaceEmbedder()
    scores, kept, undetected = [], [], 0

    for pair, label in zip(pairs, labels, strict=True):
        embeddings = []
        for image in pair:
            frame = _to_bgr(image)
            face = detector.detect_primary(frame)
            if face is None:
                break
            embeddings.append(embedder.embed(frame, face))
        if len(embeddings) != 2:
            undetected += 1
            continue
        scores.append(cosine_similarity(*embeddings))
        kept.append(int(label))

    return np.array(scores), np.array(kept), undetected


def rates_at(scores: np.ndarray, labels: np.ndarray, threshold: float) -> tuple[float, float]:
    genuine, impostor = scores[labels == 1], scores[labels == 0]
    tar = float((genuine >= threshold).mean())
    trr = float((impostor < threshold).mean())
    return tar, trr


def threshold_for_trr(scores: np.ndarray, labels: np.ndarray, target_trr: float) -> float:
    impostor = np.sort(scores[labels == 0])
    index = int(np.ceil(target_trr * len(impostor))) - 1
    index = min(max(index, 0), len(impostor) - 1)
    return float(impostor[index])


def best_accuracy_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    candidates = np.unique(scores)
    accuracies = [(float(((scores >= t) == (labels == 1)).mean()), float(t)) for t in candidates]
    accuracy, threshold = max(accuracies)
    return threshold, accuracy


TRR_TARGETS = (0.99, 0.995, 0.999, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="test", choices=["train", "test", "10_folds"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cache", default="data/lfw_scores.npz")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    cache = Path(args.cache)
    if cache.exists():
        with np.load(cache) as data:
            scores, labels, undetected = data["scores"], data["labels"], int(data["undetected"])
    else:
        scores, labels, undetected = pair_scores(args.subset, args.limit)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, scores=scores, labels=labels, undetected=undetected)

    best_threshold, best_accuracy = best_accuracy_threshold(scores, labels)
    operating_points = []
    for target in TRR_TARGETS:
        threshold = threshold_for_trr(scores, labels, target)
        tar, trr = rates_at(scores, labels, threshold)
        operating_points.append(
            {"target_trr": target, "threshold": round(threshold, 4),
             "true_accept_rate": round(tar, 4), "true_reject_rate": round(trr, 4)},
        )

    result = {
        "subset": args.subset,
        "pairs_scored": len(scores),
        "pairs_skipped_no_detection": undetected,
        "best_accuracy": round(best_accuracy, 4),
        "best_accuracy_threshold": round(best_threshold, 4),
        "operating_points": operating_points,
    }
    if args.json:
        print(json.dumps(result, indent=2))
        return

    print(f"subset                {result['subset']}")
    print(f"pairs scored          {result['pairs_scored']} "
          f"({result['pairs_skipped_no_detection']} skipped, no detection)")
    print(f"best accuracy         {best_accuracy:.4f} at threshold {best_threshold:.4f}")
    print()
    print(f"{'target TRR':>12} {'threshold':>10} {'TAR':>8} {'TRR':>8}")
    for point in operating_points:
        print(f"{point['target_trr']:>12.3f} {point['threshold']:>10.4f} "
              f"{point['true_accept_rate']:>8.4f} {point['true_reject_rate']:>8.4f}")


if __name__ == "__main__":
    main()
