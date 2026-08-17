#!/usr/bin/env python3
"""Train the 2026 RFI/scintillation HGB model and plot its confusion matrix.

This follows the earlier ``supervised_classification.py`` workflow: split the
data 80/20 by whole UTC day, duplicate the RFI training rows 20 times, fit a
histogram-based gradient-boosting classifier, and plot a row-normalized
validation confusion matrix with raw counts.
"""

# %% Settings and imports
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
)


DATA_PATH = Path(
    "/Users/isaac/Documents/ML_Analysis_2026/labeled_ml_features_2026.pq"
)
FIGURE_PATH = Path(__file__).with_name("histgb_confusion_matrix_2026.png")

RANDOM_SEED = 0
TRAIN_FRACTION = 0.80
RFI_EXTRA_COPIES = 20

# Physical signal features only. Identifiers, timestamps, station/source
# metadata, geometry, sample counts, and carrier frequencies are excluded.
FEATURES = [
    "snr1_power_db_0p017_0p05_hz",
    "snr1_power_db_0p05_0p1_hz",
    "snr1_power_db_0p1_0p3_hz",
    "snr1_power_db_0p3_1_hz",
    "snr1_power_db_1_10_hz",
    "snr2_power_db_0p017_0p05_hz",
    "snr2_power_db_0p05_0p1_hz",
    "snr2_power_db_0p1_0p3_hz",
    "snr2_power_db_0p3_1_hz",
    "snr2_power_db_1_10_hz",
    "tec12_power_db_0p017_0p05_hz",
    "tec12_power_db_0p05_0p1_hz",
    "tec12_power_db_0p1_0p3_hz",
    "tec12_power_db_0p3_1_hz",
    "tec12_power_db_1_10_hz",
    "common_delta_snr1_std_dbhz",
    "common_delta_snr1_median_dbhz",
    "common_delta_snr1_p95_dbhz",
    "common_delta_snr1_p99_dbhz",
    "common_delta_snr2_std_dbhz",
    "common_delta_snr2_median_dbhz",
    "common_delta_snr2_p95_dbhz",
    "common_delta_snr2_p99_dbhz",
    "s4_1",
    "s4_2",
]


# %% Train, evaluate, and plot
def main() -> None:
    frame = pd.read_parquet(DATA_PATH)
    required = {*FEATURES, "label", "minute_timestamp_utc"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Dataset is missing required columns: {missing}")

    frame = frame.dropna(subset=["label", "minute_timestamp_utc"]).copy()
    frame["date"] = pd.to_datetime(
        frame["minute_timestamp_utc"], errors="raise"
    ).dt.normalize()

    unique_dates = frame["date"].drop_duplicates().to_numpy()
    if len(unique_dates) < 2:
        raise ValueError("At least two unique dates are required for an 80/20 split")
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(unique_dates)

    n_train_dates = max(1, int(TRAIN_FRACTION * len(unique_dates)))
    n_train_dates = min(n_train_dates, len(unique_dates) - 1)
    train_dates = unique_dates[:n_train_dates]
    validation_dates = unique_dates[n_train_dates:]

    train = frame.loc[frame["date"].isin(train_dates)]
    validation = frame.loc[frame["date"].isin(validation_dates)]
    x_train = train[FEATURES]
    y_train = train["label"]
    x_validation = validation[FEATURES]
    y_validation = validation["label"]

    rfi_mask = y_train.eq("RFI")
    if not rfi_mask.any():
        raise ValueError("The training split contains no RFI rows to oversample")
    x_train = pd.concat(
        [x_train, *([x_train.loc[rfi_mask]] * RFI_EXTRA_COPIES)],
        ignore_index=True,
    )
    y_train = pd.concat(
        [y_train, *([y_train.loc[rfi_mask]] * RFI_EXTRA_COPIES)],
        ignore_index=True,
    )

    model = HistGradientBoostingClassifier(
        max_iter=100,
        early_stopping=True,
        random_state=RANDOM_SEED,
    )
    model.fit(x_train, y_train)
    predicted = model.predict(x_validation)

    classes = model.classes_
    matrix = confusion_matrix(y_validation, predicted, labels=classes)
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals != 0,
    )

    fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=300)
    image = ax.imshow(normalized, cmap=plt.cm.Blues, vmin=0, vmax=1)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Fraction of true class")

    locations = np.arange(len(classes))
    ax.set_xticks(locations, labels=classes)
    ax.set_yticks(locations, labels=classes)
    for true_index in range(len(classes)):
        for predicted_index in range(len(classes)):
            fraction = normalized[true_index, predicted_index]
            count = matrix[true_index, predicted_index]
            ax.text(
                predicted_index,
                true_index,
                f"{fraction:.2%}\n({count:,})",
                ha="center",
                va="center",
                color="white" if fraction >= 0.55 else "black",
            )

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Histogram-Based Gradient Boosting\nValidation confusion matrix")
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Input rows: {len(frame):,}")
    print(
        f"Training dates: {len(train_dates):,}; "
        f"validation dates: {len(validation_dates):,}"
    )
    print(f"Validation rows: {len(validation):,}")
    print(f"Model iterations: {model.n_iter_:,}")
    print(f"Accuracy: {accuracy_score(y_validation, predicted):.6f}")
    print(
        "Balanced accuracy: "
        f"{balanced_accuracy_score(y_validation, predicted):.6f}"
    )
    print("Classes:", classes.tolist())
    print("Confusion matrix (rows=true, columns=predicted):")
    print(matrix)
    print(f"Saved figure: {FIGURE_PATH}")


if __name__ == "__main__":
    main()
