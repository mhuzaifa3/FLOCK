# Flock

A face-recognition door lock for Raspberry Pi, with spoof detection
so a photograph or a phone screen cannot open the door.

Flock decides identity by comparing 128-dimensional face embeddings. It never
stores photographs. The liveness checks run first, so a spoof of an enrolled
face fails before the identity match runs at all.

## Measured accuracy

Verification accuracy on the [LFW](http://vis-www.cs.umass.edu/lfw/) test pairs.
Reproduce with `python eval/recognition.py`:

| target FMR | threshold | true-accept rate | impostor accepts | 95% FMR bound | basis |
|---|---|---|---|---|---|
| 1e-2 | 0.3057 | 0.9838 | 4 of 495 | 1.8e-2 | measured |
| 1e-3 | 0.3785 | 0.9757 | 0 of 495 | 6.0e-3 | extrapolated |
| 1e-4 | 0.4384 | 0.9675 | 0 of 495 | 6.0e-3 | extrapolated |
| 1e-5 | 0.4904 | 0.9432 | 0 of 495 | 6.0e-3 | extrapolated |
| 1e-6 | 0.5370 | 0.8824 | 0 of 495 | 6.0e-3 | extrapolated |

The shipped default targets a false-match rate of **1e-6**, the rate consumer
phone face unlock is specified at, which puts the threshold at 0.5370 and costs
12% of single-frame accepts. Set it in `flock/config.py` as
`target_false_match_rate`; `flock/calibrate.py` maps the rate to a threshold.

Read the last two columns before trusting any of this. The run scores 495
impostor pairs, so its resolution bottoms out around 1 in 500. Zero accepts in
495 comparisons bounds the true rate at 6.0e-3 with 95% confidence and no
tighter, whatever threshold produced the zero. Demonstrating 1e-6 takes roughly
three million impostor comparisons, which is a different dataset than this one.

Everything below 6.0e-3 therefore comes off a normal tail fitted to the impostor
scores, extended past the data. The fit is supported in-sample (skew 0.08,
excess kurtosis -0.06, Shapiro-Wilk p=0.24) and unvalidated out of it. A tail
that is heavier than normal in the region nobody measured would put the real
rate above the target, so treat 1e-6 as a design goal Flock aims at rather than
a property it demonstrates.

Best accuracy is 0.9909 at threshold 0.3403. The run scored 988 of 1000 pairs;
the other 12 contained a face the detector did not find.

Two things this benchmark does not cover. LFW is a one-to-one verification test,
while the door runs one-to-many search against every enrolled template, which
multiplies the per-comparison rate by the size of the roster. And LFW faces are
web photographs, not a camera at a doorway at night.

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

Flock downloads model weights on first use. The repository ships none.

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
