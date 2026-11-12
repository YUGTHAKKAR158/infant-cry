"""
Step 7 (part 3): the neural network itself.

Kept intentionally small - every layer here has a direct, understandable job,
and the whole thing is small enough to realistically convert to TensorFlow
Lite / TFLite Micro later (see project roadmap). This is NOT the kind of
"stack more layers for more accuracy" model you'd use if a laptop GPU were
the only constraint - the ESP32 target shapes every choice here.

8/16 filters were chosen after an experiment, documented in
docs/tinyml_conversion.md: this version suffers a real accuracy drop under
full-integer INT8 quantization (97.8% -> 89.6%), so a 16/32-filter version
was tried on the hypothesis that more channels would give the network more
redundancy to absorb int8 rounding error. That hypothesis was WRONG - the
larger model's INT8 accuracy was worse (79.9%), not better. Reverted back to
8/16. Kept here as a documented negative result, not silently discarded.

Architecture (input is one MFCC "image": NUM_MFCC rows x time-frames columns
x 1 channel):

    Input (40, 126, 1)
        |
    Conv2D(8 filters, 3x3) + ReLU     - learns small local time-frequency patterns
    BatchNorm                         - stabilizes/speeds up training
    MaxPool2D(2x2)                    - halves resolution, keeps strongest signal
        |
    Conv2D(16 filters, 3x3) + ReLU    - combines earlier patterns into more complex ones
    BatchNorm
    MaxPool2D(2x2)
        |
    GlobalAveragePooling2D            - collapses each feature map to one number
                                         (far fewer parameters than Flatten+Dense,
                                         which matters a lot for model size on ESP32)
        |
    Dense(16) + ReLU
    Dropout(0.3)                      - randomly zeroes activations during training
                                         to reduce overfitting on our modest dataset
        |
    Dense(2) + Softmax                - outputs [P(non_baby_cry), P(baby_cry)]
"""
from tensorflow import keras
from tensorflow.keras import layers
from src.utils.config import NUM_CLASSES


def build_model(input_shape, num_classes=NUM_CLASSES):
    model = keras.Sequential([
        keras.Input(shape=input_shape),

        layers.Conv2D(8, kernel_size=3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=2),

        layers.Conv2D(16, kernel_size=3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=2),

        layers.GlobalAveragePooling2D(),

        layers.Dense(16, activation="relu"),
        layers.Dropout(0.3),

        layers.Dense(num_classes, activation="softmax"),
    ])
    return model
