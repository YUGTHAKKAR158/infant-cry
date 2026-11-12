# Phase 4: TinyML Conversion & Quantization

All numbers below are measured on the same held-out test set (134 files,
never used in training/validation).

## What was converted

| Format | File | Purpose |
|---|---|---|
| Keras (`.keras`) | `models/baby_cry_model.keras` | Original trained model |
| TFLite float32 | `models/baby_cry_model_float32.tflite` | Same precision, lighter format - isolates "did the FORMAT change hurt accuracy" |
| TFLite dynamic-range INT8 | `models/baby_cry_model_dynamic_int8.tflite` | Weights only quantized to int8, activations stay float32 - diagnostic middle point, NOT a valid ESP32/TFLite-Micro target (still needs float arithmetic at runtime) |
| TFLite full-integer INT8 | `models/baby_cry_model_int8.tflite` | Weights AND activations quantized to int8 - the format TensorFlow Lite Micro on ESP32 actually needs (no hardware floating-point unit) |

Conversion code: `src/model/convert_tflite.py`. Evaluation code (accuracy,
per-class precision/recall/F1, confusion matrix, the source-subgroup bias
check from `docs/dataset.md`, model size, laptop-CPU inference time):
`src/model/evaluate_tflite.py`. Full numbers: `outputs/metrics/tflite_comparison.json`.

## Investigation history (kept - this was a real debugging process, not a straight line)

**1st attempt (8/16-filter model, unseeded training):** found a large INT8
accuracy gap - 97.76% (float) vs 89.55% (full-integer INT8), driven by
`non_cry` recall collapsing to 73.5%. Ruled out insufficient calibration
data as the cause (recalibrating with 3x more representative samples: zero
change). Hypothesized the model was too small (1,650 params, 8/16 filters)
for int8 rounding error to average out.

**2nd attempt (16/32-filter model):** tested the "bigger model is more
quantization-robust" hypothesis directly. Result: WORSE, not better - INT8
accuracy dropped further to 79.85%, `non_cry` recall down to 44.9%.
Hypothesis rejected; reverted to 8/16 filters (`src/model/model.py`).

**Quantization-aware training (QAT):** investigated `tensorflow-model-
optimization`. Installed successfully, but `tfmot.quantization.keras.
quantize_model()` requires the legacy Keras 2 API and broke our Keras-3-based
model the instant it was imported (confirmed with a direct test, not
assumed). Fixing this would mean migrating the entire pipeline (train/
evaluate/predict/convert, the `.keras` file format itself) to a different,
older Keras API just for this one step. Judged too high-risk/high-effort for
the benefit; abandoned, toolkit uninstalled.

**Root cause found (not model size or calibration data): a training
reproducibility bug.** `src/model/train.py` never seeded Keras's weight
initialization. Re-running the exact same code could silently produce a
different, sometimes much worse, trained model - confirmed when one retrain
attempt collapsed into a degenerate model that predicted `non_baby_cry` for
100% of inputs. Fixed with `keras.utils.set_random_seed(42)` at the top of
`train.py`. Retrained from scratch with the fix.

## Final, reproducible results (seeded training)

| Model | Accuracy | Size | `non_cry` recall | `cry_minority_source` accuracy |
|---|---|---|---|---|
| Keras / TFLite float32 | 95.52% | 11.1 KB | 93.9% | 90.9% |
| TFLite dynamic-range INT8 | 94.03% | 8.0 KB | 89.8% | 90.9% |
| TFLite full-integer INT8 | **94.03%** | 8.3 KB | 85.7% | 100% |

With reproducible training, the full-integer INT8 accuracy gap is much
smaller than first observed (~1.5 points, not ~8) - most of the originally
alarming gap traced back to training-run variance, not a fundamental
quantization limitation of this architecture. `non_cry` recall still drops
somewhat under full quantization (93.9% -> 85.7%, i.e. more false alarms),
which should still be stated plainly in any report rather than only quoting
overall accuracy - but this is a workable result for an ESP32 PoC, not a
blocking problem.

(Laptop CPU timings above are NOT representative of ESP32 latency - they
only confirm quantization didn't break anything and give a rough relative
sense; real embedded latency must be measured on the actual board in
Phase 5/6.)

## Takeaways for the report

1. Format conversion alone (Keras -> TFLite float32) is lossless by
   definition - the small numeric differences seen between runs are from
   training randomness, not the conversion step.
2. Full-integer INT8 quantization (required for ESP32 TFLite Micro) costs
   real accuracy on this task, concentrated in false alarms on non-crying
   audio - worth mentioning as a known limitation of the deployed system.
3. A meaningful chunk of what first looked like a "quantization problem"
   was actually an unseeded-training reproducibility bug. This is a useful
   methodological lesson: always rule out training variance with a fixed
   seed before attributing a result to a specific technique (quantization,
   architecture size, etc.).
4. The "bigger model = more quantization-robust" intuition was tested and
   falsified for this task/dataset size - worth including as a genuine
   negative result rather than omitting it.
