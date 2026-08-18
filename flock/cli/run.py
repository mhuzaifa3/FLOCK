"""Run the lock against a camera."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2

from flock.detect import FaceDetector
from flock.embed import FaceEmbedder
from flock.enrollment import EnrollmentStore
from flock.events import CloudWatchLogsSink, JsonlFileSink, MultiSink
from flock.lock import default_lock
from flock.modelstore import ensure_models
from flock.pipeline import ALLOW, AccessPipeline

logger = logging.getLogger("flock")


def build_sink(events_path: Path, log_group: str | None, device_id: str):
    sinks = [JsonlFileSink(events_path)]
    if log_group:
        try:
            sinks.append(CloudWatchLogsSink(log_group, device_id))
        except Exception:
            logger.exception("CloudWatch sink unavailable, continuing with local log only")
    return MultiSink(*sinks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--events", type=Path, default=Path("events/access.jsonl"))
    parser.add_argument("--log-group", default=None, help="CloudWatch Logs group for events")
    parser.add_argument("--device-id", default="flock-door")
    parser.add_argument("--gpio-pin", type=int, default=18)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ensure_models()

    store = EnrollmentStore()
    if not store.names():
        logger.error("nobody is enrolled, run flock-enroll first")
        return 1
    logger.info("enrolled: %s", ", ".join(store.names()))

    pipeline = AccessPipeline(
        detector=FaceDetector(),
        embedder=FaceEmbedder(),
        store=store,
        lock=default_lock(args.gpio_pin),
        sink=build_sink(args.events, args.log_group, args.device_id),
    )

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        logger.error("cannot open camera %s", args.camera)
        return 1

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            decision = pipeline.process(frame)
            if decision.reason == ALLOW:
                logger.info("unlocked for %s (%.3f)", decision.identity, decision.similarity)
            if not args.headless:
                cv2.putText(frame, decision.reason, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0) if decision.unlocked else (0, 0, 255), 2)
                cv2.imshow("flock", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
