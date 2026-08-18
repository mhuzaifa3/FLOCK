"""Enrol a person from a webcam or a directory of images."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from flock.detect import FaceDetector
from flock.embed import FaceEmbedder
from flock.enrollment import EnrollmentStore
from flock.modelstore import ensure_models

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def _from_directory(directory: Path, detector, embedder) -> list:
    embeddings = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        face = detector.detect_primary(frame)
        if face is None:
            print(f"no face in {path.name}, skipped")
            continue
        embeddings.append(embedder.embed(frame, face))
    return embeddings


def _from_camera(samples: int, camera: int, detector, embedder) -> list:
    capture = cv2.VideoCapture(camera)
    if not capture.isOpened():
        msg = f"cannot open camera {camera}"
        raise RuntimeError(msg)
    embeddings = []
    print(f"capturing {samples} samples, press q to stop early")
    try:
        while len(embeddings) < samples:
            ok, frame = capture.read()
            if not ok:
                break
            face = detector.detect_primary(frame)
            if face is not None:
                embeddings.append(embedder.embed(frame, face))
                x, y, w, h = face.bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{len(embeddings)}/{samples}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("enrol", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return embeddings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("--images", type=Path, help="enrol from a directory instead of a camera")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    ensure_models()
    detector, embedder = FaceDetector(), FaceEmbedder()

    if args.images:
        embeddings = _from_directory(args.images, detector, embedder)
    else:
        embeddings = _from_camera(args.samples, args.camera, detector, embedder)

    if not embeddings:
        print("no usable faces found, nothing enrolled", file=sys.stderr)
        return 1

    path = EnrollmentStore().enroll(args.name, embeddings)
    print(f"enrolled {args.name} from {len(embeddings)} samples -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
