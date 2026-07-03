from pathlib import Path
import pandas as pd

from add_elevaz import sp3_merge_lvl3


def run_sp3_pipeline(
    parquet_dir,
    sp3_dir,
    output_dir,
    rx_lat=-7.21245,
    rx_long=-35.9066,
    rx_hei=552.50323,
):
    """
    Apply SP3 merge to every Level-3 parquet file.
    """

    parquet_dir = Path(parquet_dir)
    sp3_dir = Path(sp3_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(parquet_dir.glob("*.parquet"))

    print(f"\nFound {len(parquet_files)} parquet files.\n")

    success = 0
    failed = []

    for i, parquet_file in enumerate(parquet_files, start=1):

        print("=" * 80)
        print(f"[{i}/{len(parquet_files)}] {parquet_file.name}")

        try:

            # --------------------------------------------------
            # Read only the timestamp column
            # --------------------------------------------------

            df = pd.read_parquet(
                parquet_file,
                columns=["timestamp"]
            )

            first_time = pd.to_datetime(df["timestamp"].iloc[0])

            year = first_time.year
            doy = first_time.dayofyear

            # --------------------------------------------------
            # Construct SP3 filename
            # --------------------------------------------------

            sp3_file = (
                sp3_dir /
                f"IAC0MGXFIN_{year}{doy:03d}0000_01D_05M_ORB.SP3"
            )

            if not sp3_file.exists():
                raise FileNotFoundError(
                    f"SP3 file not found:\n{sp3_file}"
                )

            print(f"Using SP3 : {sp3_file.name}")

            # --------------------------------------------------
            # Merge
            # --------------------------------------------------

            unified_df, outfile = sp3_merge_lvl3(
                parquet_file=parquet_file,
                sp3_file=sp3_file,
                output_dir=output_dir,
                rx_lat=rx_lat,
                rx_long=rx_long,
                rx_hei=rx_hei,
            )

            print(f"✓ Saved : {outfile.name}")

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

        for file in failed:
            print(file)

run_sp3_pipeline(
    parquet_dir="/home/dal674840/scratch/lvl0_parquet",
    sp3_dir="/home/dal674840/scratch",
    output_dir="/home/dal674840/scratch/lvl0_welev_parquet",
)            