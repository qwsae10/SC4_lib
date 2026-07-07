from pathlib import Path
import pandas as pd

# ------------------------------------------------------------------
# Input and output directories
# ------------------------------------------------------------------
input_dir = Path("/home/dal674840/scratch/lvl0_welev_parquet")
output_dir = Path("/home/dal674840/scratch/lvl0_welev_4hr")
output_dir.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Read all parquet files
# ------------------------------------------------------------------
files = sorted(input_dir.glob("*.parquet"))

# Group files by DOY
files_by_day = {}

for f in files:
    # Example filename:
    # mx01316a00.25__mearem_welev_lvl0_lvl0.parquet
    stem = f.name

    # DOY is characters 5:8
    doy = stem[5:8]

    files_by_day.setdefault(doy, []).append(f)

# ------------------------------------------------------------------
# Concatenate every 16 files (=4 hours)
# ------------------------------------------------------------------
FILES_PER_BLOCK = 16

for doy in sorted(files_by_day.keys()):

    day_files = sorted(files_by_day[doy])

    print(f"\nProcessing DOY {doy}")
    print(f"Found {len(day_files)} files")

    for block_num in range(0, len(day_files), FILES_PER_BLOCK):

        block_files = day_files[block_num:block_num + FILES_PER_BLOCK]

        if len(block_files) < FILES_PER_BLOCK:
            print(f"Skipping incomplete block ({len(block_files)} files)")
            continue

        dfs = []

        for f in block_files:
            print(f"  Reading {f.name}")
            dfs.append(pd.read_parquet(f))

        merged = pd.concat(dfs, ignore_index=True)

        start_hour = block_num // 4
        end_hour = start_hour + 4

        outfile = (
            output_dir
            / f"DOY{doy}_{start_hour:02d}-{end_hour:02d}UTC_4hr_lvl0.parquet"
        )

        merged.to_parquet(outfile, index=False)

        print(f"Saved {outfile.name} ({len(merged):,} rows)")