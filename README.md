# Flock

A face-recognition door lock for Raspberry Pi, with spoof detection
so a photograph or a phone screen cannot open the door.

Flock decides identity by comparing 128-dimensional face embeddings. It never
stores photographs. The liveness checks run first, so a spoof of an enrolled
face fails before the identity match runs at all.

## Measured accuracy

Verification accuracy on the [LFW](http://vis-www.cs.umass.edu/lfw/) test pairs,
the standard benchmark for this task. Reproduce with `python eval/recognition.py`:

| target true-reject rate | threshold | true-accept rate | true-reject rate |
|---|---|---|---|
| 0.990 | 0.3042 | 0.9858 | 0.9899 |
| 0.995 | 0.3153 | 0.9817 | 0.9939 |
| 0.999 | 0.3388 | 0.9817 | 0.9980 |

Best accuracy is 0.9909 at threshold 0.3403. The run scored 988 of 1000 pairs;
the other 12 contained a face the detector did not find.

The shipped default is **threshold 0.3388**, giving a 98.2% true-accept rate at
a 99.8% true-reject rate. Tune a door lock by fixing how often a stranger gets
in and accepting whatever convenience that leaves, which is why the table is
indexed by true-reject rate.

The 99.8% ceiling is a property of the benchmark, not the model: 494 pairs of
different people cannot measure a rate finer than about 1 in 500.

### Liveness

Anti-spoofing accuracy depends on your camera and on the attack you care about,
and no spoof images ship with this repository, so any number here would be
meaningless. `eval/liveness.py` measures it against frames you capture yourself,
and reports how well each check separates real faces from spoofs, so you can see
which one is doing the work.

## How it works

```
frame -> YuNet detect -> texture check -> blink check -> SFace embed -> match -> unlock
                              |                |                          |
                              +----------------+--------------------------+--> access event
```

| Stage | Implementation |
|---|---|
| Detection | YuNet CNN detector via OpenCV, 5 facial landmarks |
| Texture liveness | Laplacian variance, share of high-frequency detail, saturation spread, glare fraction |
| Blink liveness | Edge detail around each eye against a rolling baseline, dips counted as blinks |
| Identity | SFace 128-d embeddings, cosine similarity against enrolled templates |
| Unlocking | GPIO relay on a Pi, simulated lock elsewhere |
| Audit | JSON Lines locally, optionally shipped to CloudWatch Logs |

Texture detection targets a face rephotographed from a screen or a print, which
loses fine detail and changes color and glare. Blink detection targets a still
photograph, which never blinks. The two cover different attacks, so both run.

## Privacy

Flock writes no face imagery to disk at any point. Enrollment stores the mean
embedding of the samples and discards the frames. Access events record the
decision and the evidence for it, never embeddings or images, so the audit trail
can leave the device without moving biometric data. `.gitignore` excludes image
and video formats.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python scripts/fetch_models.py
```

Flock downloads model weights on first use rather than shipping them in the
repository.

## Usage

```bash
flock-enroll alice                      # 20 samples from the webcam
flock-enroll alice --images ./photos    # or from a directory
flock-lock                              # run the lock
flock-lock --log-group flock/access     # also ship events to CloudWatch Logs
```

On a Raspberry Pi, Flock drives the relay from BCM pin 18 by default
(`--gpio-pin`). Off-device it falls back to a simulated lock, so the whole
system runs and tests on a laptop with no hardware attached.

## Tests

```bash
pytest
```

The tests fake the vision stages, so the suite runs without a camera, without
model weights, and without any face data.

## Evaluation

```bash
python eval/recognition.py              # LFW verification, downloads to the sklearn cache
python eval/liveness.py                 # your own live/spoof frames
```

## License

MIT
