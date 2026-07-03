from pathlib import Path
import shutil

from scintkit.pipelines.auto import process


def run_lvl3_pipeline(
    lvl0_dir,
    lvl2_dir,
    lvl3_dir,
    mode="both",
    verbose=True,
):
    """
    Run ScintKit processing on all Level-0 parquet files.
    """

    lvl0_dir = Path(lvl0_dir)
    lvl2_dir = Path(lvl2_dir)
    lvl3_dir = Path(lvl3_dir)

    lvl2_dir.mkdir(parents=True, exist_ok=True)
    lvl3_dir.mkdir(parents=True, exist_ok=True)

    lvl0_files = sorted(lvl0_dir.glob("*_lvl0.parquet"))

    print(f"\nFound {len(lvl0_files)} Level-0 files.\n")

    outputs = process(
        [str(f) for f in lvl0_files],
        verbose=verbose,
        mode=mode,
    )

    print("\nReturned outputs:")
    print(outputs)

    if outputs is None:
        print("No output files were created.")
        return

    for outfile in outputs:

        outfile = Path(outfile)

        if not outfile.exists():
            continue

        if "_lvl2" in outfile.name:
            shutil.move(
                str(outfile),
                str(lvl2_dir / outfile.name)
            )

        elif "_lvl3" in outfile.name:
            shutil.move(
                str(outfile),
                str(lvl3_dir / outfile.name)
            )

    print("\nFinished.")

run_lvl3_pipeline(
    lvl0_dir="/home/dal674840/scratch/lvl0_welev_parquet",
    lvl2_dir="/home/dal674840/scratch/lvl2_parquet",
    lvl3_dir="/home/dal674840/scratch/lvl3_parquet",
    mode="both",
    verbose=True,
)   