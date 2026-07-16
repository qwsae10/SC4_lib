import pandas as pd

file = "/home/dal674840/scratch/20240629/scintpi3_20240629_0000_19.2235E_34.4244S_v326f_lvl0.pq"

df = pd.read_parquet(file)

print("="*80)
print("DATAFRAME SHAPE")
print("="*80)
print(df.shape)

print("\n")

print("="*80)
print("COLUMNS")
print("="*80)
print(df.columns.tolist())

print("\n")

print("="*80)
print("DTYPES")
print("="*80)
print(df.dtypes)

print("\n")

print("="*80)
print("FIRST 5 ROWS")
print("="*80)
print(df.head())

print("\n")

print("="*80)
print("NULL COUNTS")
print("="*80)
print(df.isna().sum())

print("\n")

print("="*80)
print("UNIQUE CONSTELLATIONS")
print("="*80)
print(df["cons"].unique())

print("\n")

print("="*80)
print("UNIQUE SVID SAMPLE")
print("="*80)
print(sorted(df["svid"].dropna().unique())[:20])

print("\n")

print("="*80)
print("DATETIME INFO")
print("="*80)

df["datetime"] = pd.to_datetime(df["datetime"])

print(df["datetime"].min())
print(df["datetime"].max())

print("\n")

print("="*80)
print("TIME DIFFERENCE STATISTICS")
print("="*80)

diff = (
    df.sort_values("datetime")["datetime"]
      .diff()
      .value_counts()
      .head(20)
)

print(diff)

print("\n")

print("="*80)
print("CREATE MINBIN")
print("="*80)

df["minbin"] = df["datetime"].dt.floor("1min")

print(df["minbin"].head())
print("NaNs in minbin:", df["minbin"].isna().sum())

print("\n")

print("="*80)
print("CREATE PRN")
print("="*80)

constellation_map = {
    "GPS": "G",
    "GAL": "E",
    "BDS": "C",
    "GLO": "R",
    "SBAS": "S",
    "SBS": "S",
    0: "G",
    1: "S",
    2: "E",
    3: "C",
    6: "R",
}

df["prn"] = (
    df["cons"].map(constellation_map)
    + df["svid"].astype(int).astype(str).str.zfill(2)
)

print(df[["cons", "svid", "prn"]].head(20))

print("\nNaNs in PRN:", df["prn"].isna().sum())

print("\n")

print("="*80)
print("GROUP COUNTS")
print("="*80)

counts = (
    df.groupby(["minbin", "prn"])
      .size()
      .reset_index(name="n_samples")
)

print(counts.describe())

print("\n")

print("="*80)
print("TOP 20 LARGEST SAMPLE COUNTS")
print("="*80)

print(counts.sort_values("n_samples", ascending=False).head(20))

print("\n")

if len(counts) > 0:

    row = counts.loc[counts["n_samples"].idxmax()]

    print("="*80)
    print("ROW WITH MAXIMUM SAMPLES")
    print("="*80)

    print(row)

    tmp = df[
        (df["minbin"] == row["minbin"]) &
        (df["prn"] == row["prn"])
    ].sort_values("datetime")

    print("\nFirst 20 timestamps")

    print(tmp["datetime"].head(20))

    print("\nLast 20 timestamps")

    print(tmp["datetime"].tail(20))

print("\nDone.")