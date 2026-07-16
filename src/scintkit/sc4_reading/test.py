import pandas as pd

df = pd.read_parquet(
    "/home/dal674840/scratch/20240629/scintpi3_20240629_0000_19.2235E_34.4244S_v326f_lvl0.pq"
)

df["datetime"] = pd.to_datetime(df["datetime"])
df["minbin"] = df["datetime"].dt.floor("1min")

# If prn doesn't exist yet:
constellation_map = {
    "GPS": "G",
    "GAL": "E",
    "BDS": "C",
    "GLO": "R",
    "SBAS": "S",
    "SBS": "S",
}

df["prn"] = (
    df["cons"].map(constellation_map)
    + df["svid"].astype(int).astype(str).str.zfill(2)
)

counts = (
    df.groupby(["minbin", "prn"])
      .size()
      .reset_index(name="n_samples")
)

print(counts["n_samples"].describe())
print()
print(counts.sort_values("n_samples", ascending=False).head(20))