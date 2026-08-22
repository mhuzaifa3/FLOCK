"""Measures verification accuracy on the LFW pairs benchmark.

Reports the true-accept rate at a fixed false-match rate, which is the way a
door lock is actually tuned: fix how often a stranger gets in, then see how
often the owner has to try twice.

Every operating point carries the Clopper-Pearson upper bound on its false-match
rate, because the run scores a few hundred impostor pairs and cannot resolve a
rate finer than roughly 1 in 500. A row whose target sits below that bound is
read off the fitted tail, not observed.

The roster table below the operating points shows what one-to-many search costs:
the door spends its false-match budget across every enrolled template, so the
threshold climbs and the accept rate falls as people are added.

``--fit`` prints the impostor distribution constants that flock/calibrate.py
ships, so the shipped threshold can be regenerated from a run.

The dataset is downloaded to the scikit-learn cache, not into this repository.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_lfw_pairs

from flock.calibrate import (
    IMPOSTOR_COMPARISONS,
    IMPOSTOR_MEAN,
    IMPOSTOR_STDDEV,
    comparisons_for_fmr,
    fmr_upper_bound,
    threshold_for_fmr,
)
from flock.config import DEFAULT_THRESHOLDS
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


FMR_TARGETS = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
ROSTER_SIZES = (1, 5, 10, 20, 50)


def impostor_fit(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """The constants flock/calibrate.py needs, plus the shape checks behind them."""
    impostor = scores[labels == 0]
    centred = impostor - impostor.mean()
    variance = float((centred**2).mean())
    return {
        "mean": float(impostor.mean()),
        "stddev": float(impostor.std(ddof=1)),
        "comparisons": int(impostor.size),
        "max_observed": float(impostor.max()),
        "skew": float((centred**3).mean() / variance**1.5),
        "excess_kurtosis": float((centred**4).mean() / variance**2 - 3.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="test", choices=["train", "test", "10_folds"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cache", default="data/lfw_scores.npz")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--fit", action="store_true",
                        help="print the impostor distribution constants and exit")
    args = parser.parse_args()

    cache = Path(args.cache)
    if cache.exists():
        with np.load(cache) as data:
            scores, labels, undetected = data["scores"], data["labels"], int(data["undetected"])
    else:
        scores, labels, undetected = pair_scores(args.subset, args.limit)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, scores=scores, labels=labels, undetected=undetected)

    fit = impostor_fit(scores, labels)
    if args.fit:
        print(json.dumps(fit, indent=2))
        return

    impostors = int((labels == 0).sum())
    best_threshold, best_accuracy = best_accuracy_threshold(scores, labels)
    operating_points = []
    for target in FMR_TARGETS:
        threshold = threshold_for_fmr(target)
        tar, trr = rates_at(scores, labels, threshold)
        accepts = int((scores[labels == 0] >= threshold).sum())
        operating_points.append(
            {"target_fmr": target, "threshold": round(threshold, 4),
             "true_accept_rate": round(tar, 4), "true_reject_rate": round(trr, 4),
             "impostor_accepts": accepts,
             "fmr_upper_bound": round(fmr_upper_bound(accepts, impostors), 5),
             "measured": target >= fmr_upper_bound(0, impostors)},
        )

    shipped = DEFAULT_THRESHOLDS
    roster_points = []
    for size in ROSTER_SIZES:
        threshold = shipped.match_cosine_for_roster(size)
        tar, _ = rates_at(scores, labels, threshold)
        roster_points.append(
            {"enrolled": size,
             "per_comparison_fmr": shipped.target_false_match_rate / size,
             "threshold": round(threshold, 4),
             "true_accept_rate": round(tar, 4)},
        )

    result = {
        "subset": args.subset,
        "pairs_scored": len(scores),
        "impostor_comparisons": impostors,
        "pairs_skipped_no_detection": undetected,
        "best_accuracy": round(best_accuracy, 4),
        "best_accuracy_threshold": round(best_threshold, 4),
        "measurement_floor_fmr": round(fmr_upper_bound(0, impostors), 5),
        "shipped_target_fmr": shipped.target_false_match_rate,
        "shipped_threshold": round(shipped.match_cosine, 4),
        "shipped_is_extrapolated": shipped.match_is_extrapolated,
        "impostor_fit": {k: round(v, 6) for k, v in fit.items()},
        "operating_points": operating_points,
        "roster_points": roster_points,
    }
    if args.json:
        print(json.dumps(result, indent=2))
        return

    floor = result["measurement_floor_fmr"]
    print(f"subset                {result['subset']}")
    print(f"pairs scored          {result['pairs_scored']} "
          f"({result['pairs_skipped_no_detection']} skipped, no detection)")
    print(f"impostor comparisons  {impostors}")
    print(f"best accuracy         {best_accuracy:.4f} at threshold {best_threshold:.4f}")
    print(f"measurement floor     FMR {floor:.2e} (95% bound on zero accepts)")
    print()
    print(f"{'target FMR':>12} {'threshold':>10} {'TAR':>8} {'TRR':>8} "
          f"{'accepts':>8} {'95% FMR<=':>11}  basis")
    for point in operating_points:
        basis = "measured" if point["measured"] else "extrapolated"
        print(f"{point['target_fmr']:>12.0e} {point['threshold']:>10.4f} "
              f"{point['true_accept_rate']:>8.4f} {point['true_reject_rate']:>8.4f} "
              f"{point['impostor_accepts']:>8d} {point['fmr_upper_bound']:>11.2e}  {basis}")
    print()
    print(f"shipped: FMR {shipped.target_false_match_rate:.0e} "
          f"at threshold {shipped.match_cosine:.4f}"
          + (" (extrapolated)" if shipped.match_is_extrapolated else ""))
    if shipped.match_is_extrapolated:
        needed = comparisons_for_fmr(shipped.target_false_match_rate)
        print(f"         demonstrating it needs ~{needed:,} impostor comparisons, "
              f"this run has {impostors}")
    print()
    print(f"one-to-many cost at a system FMR of {shipped.target_false_match_rate:.0e}")
    print(f"{'enrolled':>10} {'per-comparison':>15} {'threshold':>10} {'TAR':>8}")
    for point in roster_points:
        print(f"{point['enrolled']:>10d} {point['per_comparison_fmr']:>15.1e} "
              f"{point['threshold']:>10.4f} {point['true_accept_rate']:>8.4f}")

    if (fit["comparisons"], round(fit["mean"], 6)) != (IMPOSTOR_COMPARISONS, IMPOSTOR_MEAN):
        print()
        print("calibration drift: flock/calibrate.py ships "
              f"mean={IMPOSTOR_MEAN} stddev={IMPOSTOR_STDDEV} n={IMPOSTOR_COMPARISONS}, "
              f"this run fitted mean={fit['mean']:.6f} stddev={fit['stddev']:.6f} "
              f"n={fit['comparisons']}. Rerun with --fit and update the constants.")


if __name__ == "__main__":
    main()
