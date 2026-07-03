from pathlib import Path
import shutil

from scintkit.pipelines.auto import process


def run_scintkit_pipeline(
    lvl0_dir,
    lvl2_dir,
    lvl3_dir,
    verbose=True,
    mode="both",
):
    """
    Run the ScintKit pipeline on every Level-0 parquet file.
    """

    lvl0_dir = Path(lvl0_dir)
    lvl2_dir = Path(lvl2_dir)
    lvl3_dir = Path(lvl3_dir)

    lvl2_dir.mkdir(parents=True, exist_ok=True)
    lvl3_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(lvl0_dir.glob("*.parquet"))

    print(f"\nFound {len(parquet_files)} Level-0 parquet files.\n")

    success = 0
    failed = []

    for i, parquet_file in enumerate(parquet_files, start=1):

        print("=" * 80)
        print(f"[{i}/{len(parquet_files)}] {parquet_file.name}")

        try:

            outputs = process(
                str(parquet_file),
                verbose=verbose,
                mode=mode,
            )

            # Move outputs to desired folders
            for f in outputs:

                f = Path(f)

                if "lvl2" in f.name.lower():
                    shutil.move(str(f), lvl2_dir / f.name)

                elif "lvl3" in f.name.lower():
                    shutil.move(str(f), lvl3_dir / f.name)

            print("✓ Success")
            success += 1

        except Exception as e:

            print("✗ Failed")
            print(e)

            failed.append(parquet_file.name)

    print("\n" + "=" * 80)
    print("Pipeline completed")
    print("=" * 80)
    print(f"Successful : {success}")
    print(f"Failed     : {len(failed)}")

    if failed:
        print("\nFailed files:")
        for f in failed:
            print(f)

run_scintkit_pipeline(
    lvl0_dir="/home/dal674840/scratch/lvl0_welev_parquet",
    lvl2_dir="/home/dal674840/scratch/lvl2_parquet",
    lvl3_dir="/home/dal674840/scratch/lvl3_parquet",
)            