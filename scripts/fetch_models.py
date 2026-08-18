"""Download the ONNX weights Flock needs. Run once after cloning."""
import logging
import sys

from flock.modelstore import ensure_models

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    force = "--force" in sys.argv
    for path in ensure_models(force=force):
        print(f"ready: {path}")
