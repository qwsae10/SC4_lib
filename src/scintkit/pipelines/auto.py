
import os
from scintkit.preprocessing.format import temp_formating,make_1min
from scintkit.services.compute import add_products
from scintkit.services.convert_to_parquet import process_one

import pandas as pd
import os
import pandas as pd
from pathlib import Path

def get_type(f):
    name = Path(f).name.lower()

    if name.endswith(".bin.zip"):
        return "binzip"
    if name.endswith(".bin"):
        return "bin"
    if name.endswith("_lvl0.pq") or name.endswith("_lvl0.parquet"):
        return "lvl0"
    if name.endswith("_lvl1.pq") or name.endswith("_lvl1.parquet"):
        return "lvl1"
    if name.endswith("_lvl2.pq") or name.endswith("_lvl2.parquet"):
        return "lvl2"
    if name.endswith("_lvl3.pq") or name.endswith("_lvl3.parquet"):
        return "lvl3"
    elif name.endswith(".pq") or name.endswith(".parquet"):
        return "pq"
    return None


def process(flist, verbose=False):
    """
    Temporary wrapper to run full pipeline on list of files and make scintillation index product files (lvl3)
    Inputs:
    - flist: list of file paths to process. Can be .bin, .bin.zip, or .pq files. Output files will have same relative path but with _lvl3.pq suffix.
    - verbose: if True, print progress messages.

    """

    if flist is None:
        flist = []
    elif isinstance(flist, (str, os.PathLike)):
        flist = [flist]

    if isinstance(flist, (list, tuple, set)):
        flist = list(flist)
    else:
        raise TypeError(
            "invalid file list type: expected path-like, list, tuple, or set, "
            f"got {type(flist)}"
        )

    converted_files = []

    allowed_types = ['bin', 'binzip', 'lvl0',]

    flist=[f for f in flist if get_type(f) in allowed_types]
    
    for fname in flist:
        
        

        if verbose:
            print(f"Processing {fname}...")

        ext = os.path.splitext(str(fname))[1].lower()

        # skip conversion if already parquet
        if ext in [".pq", ".parquet"]:
            pq_fname = fname
        else:
            pq_fname = process_one(fname)

        if verbose:
            print(f"Reading and formatting parquet file: {pq_fname}...")
        df = pd.read_parquet(pq_fname)
        df = temp_formating(df)

        df = add_products(df, verbose=verbose)
        df = make_1min(df)

        outname = str(pq_fname).replace(".pq", "_lvl3.pq").replace(".parquet", "_lvl3.pq")

        df.to_parquet(outname)
        converted_files.append(outname)

        if verbose:
            print(f"Finished processing {fname}.")

    return converted_files