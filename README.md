# Flock

A face-recognition door lock for Raspberry Pi, with presentation-attack detection
so a photograph or a phone screen cannot open the door.

Identity is decided by comparing 128-dimensional face embeddings, not by storing
photographs. Liveness is checked before identity, so a spoof of an enrolled face
is rejected without the identity ever being considered.

## Measured accuracy

Verification accuracy on the [LFW](http://vis-www.cs.umass.edu/lfw/) test pairs,
the standard benchmark for this task. Reproduce with `python eval/recognition.py`:

| target TRR | threshold | TAR | TRR |
|---|---|---|---|
| 0.990 | 0.3042 | 0.9858 | 0.9899 |
| 0.995 | 0.3153 | 0.9817 | 0.9939 |
| 0.999 | 0.3388 | 0.9817 | 0.9980 |

Best accuracy is 0.9909 at threshold 0.3403. 988 of 1000 pairs were scored; the
other 12 contained a face the detector did not find.

The shipped default is **threshold 0.3388**, giving a 98.2% true-accept rate at a
99.8% true-reject rate. A door lock should be tuned by fixing how often a stranger
gets in and accepting whatever convenience that leaves, which is why the table is
indexed by true-reject rate.

The 99.8% ceiling is a property of the benchmark, not the model: 494 impostor pairs
cannot resolve a rate finer than about 1 in 500.

### Liveness

Anti-spoofing accuracy is **not** reported here, because it depends on your camera
and on the attack you care about, and no spoof corpus ships with this repository.
`eval/liveness.py` measures it against frames you capture yourself, and prints the
per-cue separation so you can see which signal is carrying the decision.

## How it works

```
frame -> YuNet detect -> texture check -> blink check -> SFace embed -> match -> unlock
                              |                |                          |
                              +----------------+--------------------------+--> access event
```

| Stage | Implementation |
|---|---|
| Detection | YuNet CNN detector via OpenCV, 5 facial landmarks |
| Texture liveness | Laplacian variance, spectral high-frequency ratio, saturation spread, specular fraction |
| Blink liveness | Per-eye gradient energy tracked against a rolling baseline, dips counted as blinks |
| Identity | SFace 128-d embeddings, cosine similarity against enrolled templates |
| Actuation | GPIO relay on a Pi, simulated lock elsewhere |
| Audit | JSON Lines locally, optionally shipped to CloudWatch Logs |

Texture detection targets a recapture from a screen or print, which loses fine
detail and shifts colour and specular statistics. Blink detection targets a still
photograph, which never blinks. The two cover different attacks, so both run.

## Privacy

No face imagery is written to disk at any point. Enrolment stores the mean
embedding of the samples and discards the frames. Access events record the
decision and the evidence for it, never embeddings or images, so the audit trail
can leave the device without moving biometric data. `.gitignore` excludes image
and video formats outright.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python scripts/fetch_models.py
```

Model weights are downloaded on first use rather than committed.

## Usage

```bash
flock-enroll alice                      # 20 samples from the webcam
flock-enroll alice --images ./photos    # or from a directory
flock-lock                              # run the lock
flock-lock --log-group flock/access     # also ship events to CloudWatch Logs
```

On a Raspberry Pi the relay is driven from BCM pin 18 by default (`--gpio-pin`).
Off-device, a simulated lock is used automatically, so the whole system runs and
is testable on a laptop with no hardware attached.

## Tests

```bash
pytest
```

The vision stages are faked in tests, so the suite runs without a camera, without
model weights, and without any face data.

## Evaluation

```bash
python eval/recognition.py              # LFW verification, downloads to the sklearn cache
python eval/liveness.py                 # your own live/spoof frames
```

## Licence

MIT
