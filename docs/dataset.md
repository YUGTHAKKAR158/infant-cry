# Dataset Documentation — Baby Cry Detection PoC

## Source

- **Name:** Infant Cry Dataset
- **Kaggle URL:** https://www.kaggle.com/datasets/sanmithasadhish/infant-cry-dataset
- **License:** TODO — could not be scraped automatically (Kaggle page is JS-rendered).
  Please open the dataset page, check the "Usability"/license panel, and paste the
  exact license text here before this dataset is used in any report or shared publicly.
- **Downloaded to:** `data/raw/infant-cry-dataset/Dataset/`
- **Date downloaded:** 2026-09-09 (per task context)

## Raw structure as downloaded

```
Dataset/
├── data.csv        # 889 rows, columns: is_cry (yes/no), filename — label index, matches file counts
├── cry/             565 files
└── not_cry/         324 files
```

## Measured properties (via `soundfile`, every file, not a sample)

| | `cry/` | `not_cry/` |
|---|---|---|
| File count | 565 | 324 |
| Sample rates | 8000 Hz: 457 files · 44100 Hz: 108 files | 44100 Hz: 324 files |
| Channels | mono (1) — all files | mono (1) — all files |
| Encodings | PCM_16: 457 · VORBIS (.ogg): 40 · FLOAT (.wav): 68 | VORBIS (.ogg): 108 · FLOAT (.wav): 216 |
| Duration | min 4.96s, avg 6.55s, max 7.06s | min 5.00s, avg 5.00s, max 5.02s (essentially fixed) |

Class balance: 565 / 324 ≈ **63.5% / 36.5%** (~1.7 : 1). Present but not severe — needs
class weighting / stratified metrics during training, not resampling.

## Important finding: format is confounded with label

The 457 PCM16 8kHz ~7s files in `cry/` are recognizable as the **Donate-a-Cry corpus**
signature. They make up 81% of the cry class. **Zero** `not_cry` files share that
signature — 100% of `not_cry` is 44.1kHz, mostly exactly 5.00s.

This means sample rate / encoding / clip length are, on their own, almost enough to
separate the classes without listening to the actual sound. A CNN trained on raw
resampled audio could learn to key off left-over artifacts of that difference
(e.g. the 8kHz-native files have no energy above ~4kHz once resampled, since their
original Nyquist limit was 4kHz) rather than genuine cry acoustic features. That would
look like great laptop accuracy and fail completely on new real-world audio — which is
exactly the failure mode this project needs to avoid before trusting any reported number.

**Decided mitigation** (confirmed with project owner, to implement in preprocessing):
1. Resample everything to one common rate (needed anyway for consistent features).
2. Apply a low-pass filter at ~4 kHz to *all* clips (both classes) before feature
   extraction, so the 8kHz-native subset's limited bandwidth is no longer a unique
   "tell" — every clip is capped at the same effective bandwidth. This is not a purely
   artificial fix either: a real ESP32 + I2S mic setup won't capture pristine
   full-bandwidth audio either, so band-limiting is a reasonable step toward the
   eventual embedded pipeline anyway.
3. After training, separately evaluate accuracy on just the 108 44.1kHz-native `cry`
   files vs `not_cry`, as a check that the model isn't only doing well because of the
   dominant 8kHz subset's obvious cue.

## Standardization target (decided)

| Parameter | Value | Rationale |
|---|---|---|
| Sample rate | 16000 Hz | Common TinyML/speech default; above the ~4kHz low-pass cutoff with headroom. |
| Channels | mono | All source files are already mono. |
| Duration | 2.0 seconds (trim/pad) | Balance between capturing a cry's pitch contour and staying light enough for eventual real-time ESP32 inference. |
| Low-pass filter | ~4000 Hz cutoff, applied to all clips | Removes the sample-rate/format confound described above; also realistic for a real microphone. |

## Pipeline bug found and fixed during visualization

Initial implementation cropped long clips by taking `waveform[:num_samples]`
(the first 2 seconds). Measuring RMS energy across a full original 7s `cry`
clip (`203c.wav`) showed 0.0-1.5s was near-silent lead-in, with the actual
crying concentrated between 1.6-6.2s. The naive crop would have kept mostly
silence and discarded most of the real cry signal for many clips.

**Fix:** `crop_to_most_energetic_window()` in `src/audio/preprocessing.py`
slides a 2-second window across the full clip in 0.1s steps and keeps the
position with the highest total energy, instead of always taking the start.
Verified visually afterward (`outputs/plots/baby_cry_203c.png`,
`baby_cry_123c.png`) — the cropped window now contains continuous,
structured cry bursts across the full 2 seconds instead of silence.

## Bias check result (after first training run)

As planned, test accuracy was broken down by the cry class's original source
subgroup (native 8kHz "Donate-a-Cry" signature vs native 44.1kHz):

| Subgroup | n (test) | Accuracy |
|---|---|---|
| Cry, 8kHz-native | 74 | 97.3% |
| Cry, 44.1kHz-native | 11 | 45.5% |
| Non-cry | 49 | 100% |

Overall test accuracy was 94.0%, but that number is dominated by the 8kHz-
native subgroup, which is also 74% of training data (457/622 train examples).
The model performs far worse on cry audio from the minority source, despite
the low-pass-filter mitigation already applied - suggesting it is still
partly learning source-specific characteristics (background noise, mic
distance/quality, session acoustics) rather than purely generic cry acoustics.
Sample size for the minority subgroup is small (11), so this is suggestive
rather than definitive, but it matches the exact risk flagged before training
and means the headline 94% should NOT be reported/trusted as the model's
real-world generalization accuracy without further work.

**Mitigation applied:** oversampled the minority (44.1kHz-native) cry
training files with 4 augmented copies each (`src/audio/augmentation.py`:
random time-shift, gain, and added noise - applied ONLY to the training
split, never to validation/test, so evaluation numbers stay honest). See
`src/model/train.py::build_minority_augmented_examples`.

**Result after mitigation** (same held-out test set, model retrained):

| Subgroup | n (test) | Accuracy before | Accuracy after |
|---|---|---|---|
| Cry, 8kHz-native | 74 | 97.3% | 100% |
| Cry, 44.1kHz-native | 11 | 45.5% | **81.8%** |
| Non-cry | 49 | 100% | 100% |
| Overall | 134 | 94.0% | 98.5% |

The minority subgroup improved substantially (45.5% -> 81.8%), and overall
test accuracy also rose to 98.5%. Still 2/11 misses in the smallest,
noisiest subgroup - not perfect, and 11 examples is too small to be fully
confident in the exact percentage - but this is a genuine, honest
improvement, not just a headline number. Getting more real (not just
augmented) cry audio from diverse sources remains valuable future work
before claiming strong real-world generalization.

## Phase 3: general augmentation + SpecAugment (further improvement)

Extended the mitigation above in two ways (`src/model/train.py`,
`src/audio/augmentation.py`):
1. Every training file (not just the minority subgroup) now gets 1 general
   augmented copy (time-shift/gain/noise), for broader robustness.
2. Added SpecAugment (random frequency-band + time-span masking applied
   directly to the MFCC) to every augmented copy, to reduce overfitting risk
   given how small this dataset is.

Result on the same held-out test set:

| Subgroup | n | Before (Phase 3) | After (Phase 3) |
|---|---|---|---|
| Cry, dominant source | 74 | 100% | 98.6% |
| Cry, minority source | 11 | 81.8% | **90.9%** |
| Non-cry | 49 | 100% | 98.0% |
| Overall | 134 | 98.5% | 97.8% |

The minority subgroup (the one previously showing the real generalization
gap) improved further; overall accuracy shifted slightly down by one example
elsewhere, which is expected/acceptable noise at this test-set size (134
examples) and consistent with the model relying less on dominant-subgroup
shortcuts.

**Note (superseded by a reproducibility fix in Phase 4):** the 97.8%/90.9%
numbers above came from a run where Keras's weight initialization was not
seeded. While investigating Phase 4 quantization results, this
non-reproducibility was found to be a real bug (one retrain attempt
collapsed into a degenerate model) and fixed with
`keras.utils.set_random_seed(42)` in `src/model/train.py`. The seeded
re-run's official numbers are in `docs/tinyml_conversion.md`
(95.52% overall / 90.9% minority-subgroup) - that version, not this one, is
the one saved to `models/` and carried forward.

## Not yet measured / TODO

- License text (see above).
- Per-file amplitude/loudness distribution.
- Whether `not_cry` clips are silence, room noise, or other environmental/human
  sounds (need to actually listen to a sample).
- Speaker/session metadata — `data.csv` only has `is_cry` + `filename`, no
  age/gender/source-session info, so we cannot control for a given
  recording/session appearing in both train and test splits beyond avoiding
  duplicate filenames.
