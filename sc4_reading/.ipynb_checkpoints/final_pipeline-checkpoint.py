import os
from pathlib import Path
import pandas as pd
from bin_to_parquet import run_pipeline
from add_elevaz_pipeline import run_sp3_pipeline
from lvl3_pipeline import run_lvl3_pipeline

def get_input():
    '''Get user input for the binary files'''
    binary_dir = input("Enter the path to the binary files:")
    latitude = float(input("Enter the latitude of the receiver: "))
    longitude = float(input("Enter the longitude of the receiver: "))
    height = float(input("Enter the height of the receiver: "))
    return binary_dir, latitude, longitude, height

def main():
    binary_dir, latitude, longitude, height = get_input()

    binary_dir = Path(binary_dir)
    txt_dir = binary_dir / "txt"
    mearem_dir = binary_dir / "mearem"
    lvl0_dir = binary_dir / "lvl0" 
    sp3_dir = binary_dir / "sp3"
    lvl2_dir = binary_dir / "lvl2"
    lvl3_dir = binary_dir / "lvl3"  

    print(f"\nBinary directory: {binary_dir}")

    run_pipeline(
        binary_dir=binary_dir,
        txt_dir=txt_dir,
        mearem_dir=mearem_dir,
        lvl0_dir=lvl0_dir
    )

    print("Binary to Parquet pipeline completed.")

    run_sp3_pipeline(
        parquet_dir=lvl0_dir,
        sp3_dir=sp3_dir,
        output_dir=lvl2_dir,
        rx_lat=latitude,
        rx_long=longitude,
        rx_hei=height
    )

    print("Elevation and Azimuth added to the parquet files.")

    run_lvl3_pipeline(
        lvl0_dir=lvl0_dir,
        lvl2_dir=lvl2_dir,
        lvl3_dir=lvl3_dir,
        mode="both",
        verbose=True
    )

    print("Level-3 pipeline completed.")

if __name__ == "__main__":
    main()

