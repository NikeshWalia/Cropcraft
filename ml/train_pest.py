"""Train the pest classifier by fine-tuning MobileNetV2 on the clean splits.

Differences from ``cnn_model.py`` in CropCraft 1.x:

*   **Preprocessing is baked into the saved model.** 1.x rescaled by 1/255
    during training but fed raw 0-255 pixels at inference, so the deployed
    model saw inputs it was never trained on. Here the normalisation layer is
    part of the graph, so training and serving cannot diverge.
*   **Transfer learning instead of a 3-layer CNN from scratch.** The honest
    dataset is ~70 images per class, far too little to learn features from
    nothing.
*   **Real validation data.** 1.x validated on a set that was 88% copied from
    its own training data, and set ``validation_steps=6500`` against ~16
    batches, looping the generator hundreds of times per epoch.
*   **Early stopping on validation loss** rather than a fixed 100 epochs.
*   **The test split is scored once**, after training finishes.

Run ``python -m ml.prepare_dataset`` first, then::

    python -m ml.train_pest
"""

from __future__ import annotations

import json
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "Data" / "clean"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "pest_classifier.keras"
LABELS_PATH = MODEL_DIR / "pest_labels.json"
METRICS_PATH = MODEL_DIR / "pest_metrics.json"

IMAGE_SIZE = 224
BATCH_SIZE = 32
SEED = 42
HEAD_EPOCHS = 30
FINETUNE_EPOCHS = 25


def load_split(name: str, shuffle: bool) -> tf.data.Dataset:
    """Load one split as a batched dataset of raw 0-255 images."""
    return keras.utils.image_dataset_from_directory(
        DATA_ROOT / name,
        labels="inferred",
        label_mode="categorical",
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED,
    )


def build_model(n_classes: int) -> tuple[keras.Model, keras.Model]:
    """Build the classifier. Returns ``(model, base)`` so the base can be unfrozen."""
    augment = keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.15),
            keras.layers.RandomZoom(0.15),
            keras.layers.RandomContrast(0.15),
        ],
        name="augmentation",
    )

    base = keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False

    inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="image")
    x = augment(inputs)
    # Baked-in normalisation: callers pass raw 0-255 pixels and cannot get
    # this wrong the way the 1.x inference path did.
    x = keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="preprocess")(x)
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(n_classes, activation="softmax", name="pest")(x)

    return keras.Model(inputs, outputs, name="pest_classifier"), base


def main() -> None:
    if not DATA_ROOT.is_dir():
        raise SystemExit("Data/clean is missing -- run `python -m ml.prepare_dataset` first")

    train_ds = load_split("train", shuffle=True)
    val_ds = load_split("val", shuffle=False)
    test_ds = load_split("test", shuffle=False)
    labels = train_ds.class_names
    print(f"\nclasses ({len(labels)}): {labels}\n")

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(256, seed=SEED).prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)
    test_ds = test_ds.cache().prefetch(autotune)

    model, base = build_model(len(labels))

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.4, patience=4, min_lr=1e-6, verbose=1
        ),
    ]

    # --- phase 1: train the classification head only ------------------------
    print("=" * 62)
    print("phase 1/2  frozen backbone, training head")
    print("=" * 62)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    history_head = model.fit(
        train_ds, validation_data=val_ds, epochs=HEAD_EPOCHS, callbacks=callbacks, verbose=2
    )

    # --- phase 2: fine-tune the top of the backbone -------------------------
    print("\n" + "=" * 62)
    print("phase 2/2  unfrozen top layers, fine-tuning")
    print("=" * 62)
    base.trainable = True
    for layer in base.layers[:-40]:
        layer.trainable = False
    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    history_ft = model.fit(
        train_ds, validation_data=val_ds, epochs=FINETUNE_EPOCHS, callbacks=callbacks, verbose=2
    )

    # --- score the held-out split once --------------------------------------
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
    probabilities = model.predict(test_ds, verbose=0)
    predicted = probabilities.argmax(axis=1)
    actual = np.concatenate([y.numpy().argmax(axis=1) for _, y in test_ds])

    top3 = float(
        np.mean(
            [a in row for a, row in zip(actual, probabilities.argsort(axis=1)[:, -3:], strict=True)]
        )
    )
    per_class = {
        label: round(float((predicted[actual == i] == i).mean()), 3)
        for i, label in enumerate(labels)
        if (actual == i).any()
    }

    print(f"\nheld-out accuracy       : {test_accuracy:.4f}")
    print(f"held-out top-3 accuracy : {top3:.4f}")
    print(f"held-out loss           : {test_loss:.4f}\n")
    for label, score in sorted(per_class.items(), key=lambda kv: kv[1]):
        print(f"  {label:14s} recall {score:.3f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    LABELS_PATH.write_text(json.dumps(labels, indent=2), encoding="utf-8")
    METRICS_PATH.write_text(
        json.dumps(
            {
                "backbone": "MobileNetV2 (ImageNet)",
                "image_size": IMAGE_SIZE,
                "seed": SEED,
                "epochs_head": len(history_head.history["loss"]),
                "epochs_finetune": len(history_ft.history["loss"]),
                "holdout_accuracy": round(float(test_accuracy), 4),
                "holdout_top3_accuracy": round(top3, 4),
                "holdout_loss": round(float(test_loss), 4),
                "per_class_recall": per_class,
                "keras_version": keras.__version__,
                "tensorflow_version": tf.__version__,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nsaved {MODEL_PATH}")
    print(f"saved {LABELS_PATH}")
    print(f"saved {METRICS_PATH}")


if __name__ == "__main__":
    main()
