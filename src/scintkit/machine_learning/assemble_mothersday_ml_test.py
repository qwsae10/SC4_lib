#!/usr/bin/env python3
"""Assemble a labeled May 10-11 Mother’s Day ML feature Parquet."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from scintkit.machine_learning.filename_metadata import parse_filename_coordinates


# Edit only these paths if needed.
FEATURE_DIR = Path(
    "/titan/frodrigues/mothersday/ml_features_may10_11_s4gt0p3"
)
OUTPUT_FILE = Path(
    "/titan/frodrigues/mothersday/mothersday_ml_test_20240510_20240511.pq"
)
OVERWRITE = False

START_UTC = pd.Timestamp("2024-05-10 00:00:00", tz="UTC")
END_UTC = pd.Timestamp("2024-05-12 00:00:00", tz="UTC")
MULTIPATH_START_UTC = pd.Timestamp("2024-05-10 18:00:00", tz="UTC")
S4_MIN = 0.3
COORDINATE_TOLERANCE_DEG = 0.01

# Centers chosen from the actual Mother’s Day level-2 filenames.
SITES = {
    "COL": (38.381, -103.156),
    "HND": (14.094, -87.160),
    "MOS": (38.919, -92.128),
}

REQUIRED_COLUMNS = {
    "prn",
    "svid",
    "s4_1",
    "elevation_deg",
    "minute_timestamp_utc",
}


def identify_station(path: Path) -> tuple[str | None, float, float]:
    latitude, longitude = parse_filename_coordinates(path)
    matches = [
        station
        for station, (target_latitude, target_longitude) in SITES.items()
        if abs(latitude - target_latitude) <= COORDINATE_TOLERANCE_DEG
        and abs(longitude - target_longitude) <= COORDINATE_TOLERANCE_DEG
    ]
    if len(matches) > 1:
        raise ValueError(f"ambiguous station coordinates in {path.name}: {matches}")
    return (matches[0] if matches else None), latitude, longitude


def select_and_label(frame: pd.DataFrame, feature_file: Path) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise KeyError(f"{feature_file.name} is missing {sorted(missing)}")

    station, latitude, longitude = identify_station(feature_file)
    if station is None or frame.empty:
        return frame.iloc[0:0].copy()

    timestamp = pd.to_datetime(
        frame["minute_timestamp_utc"], errors="raise", utc=True
    )
    svid = pd.to_numeric(frame["svid"], errors="coerce")
    elevation = pd.to_numeric(frame["elevation_deg"], errors="coerce")
    s4 = pd.to_numeric(frame["s4_1"], errors="coerce")

    selected = (
        timestamp.ge(START_UTC)
        & timestamp.lt(END_UTC)
        & svid.notna()
        & svid.ne(255)
        & elevation.notna()
        & elevation.le(90)
        & s4.notna()
        & s4.gt(S4_MIN)
    )

    if station == "COL":
        selected &= elevation.gt(30)
        label = "RFI"
    elif station == "HND":
        selected &= elevation.gt(20)
        label = "Scintillation"
    elif station == "MOS":
        selected &= elevation.gt(20) & timestamp.ge(MULTIPATH_START_UTC)
        label = "Multipath"
    else:  # Defensive: all SITES entries must have a rule.
        raise AssertionError(f"no label rule for {station}")

    result = frame.loc[selected].copy()
    result["station"] = station
    result["station_latitude_deg"] = latitude
    result["station_longitude_deg"] = longitude
    result["label"] = label
    return result


def keep_best_duplicate(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Keep the most complete row when restart chunks overlap a minute."""

    key = ["station", "prn", "minute_timestamp_utc"]
    duplicate_count = int(frame.duplicated(key, keep=False).sum())
    if duplicate_count == 0:
        return frame, 0

    rank = pd.Series(0.0, index=frame.index)
    for column in ("n_samples", "n_snr2_samples", "n_tec12_samples"):
        if column in frame:
            rank += pd.to_numeric(frame[column], errors="coerce").fillna(-1)
    ranked = frame.assign(_completeness_rank=rank)
    sort_columns = [*key, "_completeness_rank"]
    ascending = [True, True, True, False]
    if "source_filename" in ranked:
        sort_columns.append("source_filename")
        ascending.append(True)
    ranked = ranked.sort_values(sort_columns, ascending=ascending, kind="stable")
    ranked = ranked.drop_duplicates(key, keep="first")
    return ranked.drop(columns="_completeness_rank"), duplicate_count


def main() -> None:
    feature_dir = FEATURE_DIR.expanduser().resolve()
    output_file = OUTPUT_FILE.expanduser().resolve()
    if not feature_dir.is_dir():
        raise FileNotFoundError(feature_dir)
    if output_file.exists() and not OVERWRITE:
        raise FileExistsError(
            f"{output_file} already exists; set OVERWRITE = True to replace it"
        )

    feature_files = sorted(feature_dir.glob("*_ml_features.pq"))
    if not feature_files:
        raise FileNotFoundError(f"no ML feature Parquets in {feature_dir}")

    pieces: list[pd.DataFrame] = []
    for number, feature_file in enumerate(feature_files, start=1):
        selected = select_and_label(pd.read_parquet(feature_file), feature_file)
        if not selected.empty:
            pieces.append(selected)
        if number % 10 == 0 or number == len(feature_files):
            print(f"Read {number}/{len(feature_files)} feature files", flush=True)

    if not pieces:
        raise RuntimeError("no rows matched the requested label rules")

    labeled = pd.concat(pieces, ignore_index=True, sort=False)
    labeled, duplicate_rows = keep_best_duplicate(labeled)
    labeled = labeled.sort_values(
        ["minute_timestamp_utc", "station", "prn"], kind="stable"
    ).reset_index(drop=True)

    # Final invariants for the delivered test dataset.
    elevation = pd.to_numeric(labeled["elevation_deg"], errors="raise")
    svid = pd.to_numeric(labeled["svid"], errors="raise")
    s4 = pd.to_numeric(labeled["s4_1"], errors="raise")
    timestamp = pd.to_datetime(labeled["minute_timestamp_utc"], utc=True)
    assert svid.ne(255).all()
    assert elevation.le(90).all()
    assert s4.gt(S4_MIN).all()
    assert timestamp.ge(START_UTC).all() and timestamp.lt(END_UTC).all()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_file.with_name(
        f".{output_file.name}.in_progress_{os.getpid()}.pq"
    )
    try:
        labeled.to_parquet(
            temporary,
            index=False,
            engine="pyarrow",
            compression="zstd",
        )
        os.replace(temporary, output_file)
    finally:
        temporary.unlink(missing_ok=True)

    print(f"Wrote {len(labeled):,} rows x {len(labeled.columns)} columns")
    print(f"Output: {output_file}")
    print(f"Rows participating in duplicate keys before cleanup: {duplicate_rows:,}")
    print(labeled.groupby(["station", "label"]).size().to_string())
    missing_labels = {"RFI", "Scintillation", "Multipath"}.difference(
        labeled["label"].unique()
    )
    if missing_labels:
        print(f"WARNING: no selected rows for labels: {sorted(missing_labels)}")


if __name__ == "__main__":
    main()
