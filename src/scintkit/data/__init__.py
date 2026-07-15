"""Shared reference data used across ScintKit."""

from importlib import resources

import pandas as pd


def load_station_codes() -> pd.DataFrame:
    """Load the maintained ScintPi station registry bundled with ScintKit."""
    csv_resource = resources.files(__package__).joinpath(
        "station_scintpi_codes.csv"
    )
    with csv_resource.open("r", encoding="latin1", newline="") as csv_file:
        return pd.read_csv(csv_file)


__all__ = ["load_station_codes"]
