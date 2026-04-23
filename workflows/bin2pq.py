#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 10:04:58 2025

@author: isaac
"""

import glob 
import zipfile
import os
import re
import struct
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from multiprocessing import cpu_count,Pool
from concurrent.futures import ProcessPoolExecutor
import shutil
import tempfile


"""

This script processes .bin.zip files from the scintpi dataset,
 extracts the .bin, reads it according to its version (v324, v325, v326), 
 converts GPS week/towe to datetime, and saves as .pq in a structured directory on /titan. 
 It uses local temporary storage for processing to avoid I/O bottlenecks on shared storage.

"""
#flist_full = sorted(glob.glob("/mfs/io/groups/uars/scintpi/*/*.bin.zip"))


#ziplist = glob.glob('/mfs/io/groups/uars/scintpi/*/*.bin.zip')

#pqlist = {
#    g.replace('/mfs/io/groups/uars/scintpi/', '/titan/frodrigues/scintpi_storage/').replace('.bin.zip', '.pq')
#    for g in ziplist
#}

#real_pq_list = set(glob.glob('/titan/frodrigues/scintpi_storage/*/*.pq'))

#flist_full = [d for d in pqlist if d not in real_pq_list and (('v324' in d) | ('v325' in d )|('v326' in d))]

import glob



import glob

ziplist = glob.glob('/mfs/io/groups/uars/scintpi/*/*2024*.bin.zip')


#ziplist=glob.glob('/mfs/io/groups/uars/scintpi/sc018/*20241010*2000*19*')
pqlist = {
    g.replace('/mfs/io/groups/uars/scintpi/', '/titan/frodrigues/scintpi_storage/').replace('.bin.zip', '.pq')
    for g in ziplist
}

real_pq_list = set(glob.glob('/titan/frodrigues/scintpi_storage/*/*.pq'))


expected_pqlist=set([d for d in pqlist if (('v324' in d) | ('v325' in d )|('v326' in d))])

flist_full = expected_pqlist - real_pq_list


flist_full=[f.replace( '/titan/frodrigues/scintpi_storage/','/mfs/io/groups/uars/scintpi/').replace('.pq', '.bin.zip') for f in flist_full]


print(len(flist_full))

task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
#N = int(os.environ.get("SLURM_ARRAY_TASK_MAX", 11)) + 1  # total tasks
N=24
chunk = np.array_split(flist_full, N)[task_id]
flist = list(chunk)


def readv324(path):
    cols2read = [0, 1, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15]
    colnames = ['week', 'towe', 'cons', 'svid', 'elev', 'azim',
                'snr1', 'snr2', 'cph1', 'cph2', 'rng1', 'rng2']
    df = pd.read_csv(path, sep=r'\s+', header=None, usecols=cols2read)
    df.columns = colnames
    return df

def readv325(path):
    dt = np.dtype([
        ('week', np.int32), ('towe', np.float32),
        ('leap', np.uint8), ('cons', np.uint8),
        ('sats', np.uint8), ('svid', np.uint8),
        ('elev', np.int8), ('azim', np.int32),
        ('snr1', np.uint8), ('snr2', np.uint8),
        ('snr3', np.uint8), ('pst1', np.uint8),
        ('pst2', np.uint8), ('pst3', np.uint8),
        ('rst1', np.uint8), ('rst2', np.uint8),
        ('rst3', np.uint8), ('cph1', np.float64),
        ('cph2', np.float64), ('cph3', np.float64),
        ('rng1', np.float64), ('rng2', np.float64),
        ('rng3', np.float64), ('lon', np.float32),
        ('lat', np.float32), ('hei', np.float32),
    ], align=True)
    arr = np.fromfile(path, dtype=dt)
    return pd.DataFrame.from_records(arr)

def readv326(path):
    rec_dt = np.dtype([
       ('towe', np.float32), ('cons', np.uint8),
       ('sats', np.uint8), ('svid', np.uint8),
       ('elev', np.int8), ('azim', np.int32),
       ('snr1', np.uint8), ('snr2', np.uint8),
       ('pst1', np.uint8), ('pst2', np.uint8),
       ('rst1', np.uint8), ('rst2', np.uint8),
       ('cph1', np.float64), ('cph2', np.float64),
       ('rng1', np.float64), ('rng2', np.float64),
       ('lck1', np.int32), ('lck2', np.int32),
    ], align=True)
    hdr_size = 60
    with open(path, 'rb') as f:
        buf = f.read(hdr_size)
    fmt = '@fBbiBBBBBBddddi'
    vals = struct.unpack(fmt, buf)
    week = vals[-1]
    rec = np.fromfile(path, dtype=rec_dt, offset=64*2)
    df = pd.DataFrame.from_records(rec)
    df['week'] = week
    return df


def getvers(s):
    m = re.search(r'_v(\d+)', s)
    if m:
        return f"v{m.group(1)}"
    return None

def process_one(f):
    local_tmpdir=None
    local_zip=None
    local_bin=None
    local_pq=None
    ver=None
    try:
        # Build output parquet path (final destination on /titan)
        outzip = f.replace('/mfs/io/groups/uars/scintpi/', '/titan/frodrigues/scintpi_storage/')
        pq_location = outzip.replace('.bin.zip', '.pq')
        ver = getvers(outzip)

        # Skip if final parquet already exists
        if os.path.exists(pq_location):
            print(pq_location + ' already exists. Skipping')
            return

        # ---- LOCAL TEMP SETUP ----
        # Make a unique temp dir inside /tmp
        local_tmpdir = tempfile.mkdtemp(prefix="scintpi_", dir="/tmp")
        # Local paths
        local_zip = os.path.join(local_tmpdir, os.path.basename(f))
        local_bin = local_zip.replace('.bin.zip', '.bin')
        local_pq = local_bin.replace('.bin', '.pq')  # temp parquet in /tmp

        print(f"[LOCAL STAGE] Copy {f} -> {local_zip}")
        shutil.copy2(f, local_zip)

        # ---- UNZIP LOCALLY ----
        print(f"[UNZIP] Extracting locally -> {local_bin}")
        with zipfile.ZipFile(local_zip, 'r') as z:
            z.extractall(local_tmpdir)  # Extracts .bin here

        if not os.path.exists(local_bin):
            print(f"!! No .bin extracted in {local_tmpdir}")
            return

        # ---- PROCESS LOCALLY ----
        print(f"[READ] Reading {ver} from local {local_bin}")
        if ver == 'v324':
            df = readv324(local_bin)
        elif ver == 'v325':
            df = readv325(local_bin)
        elif ver == 'v326':
            df = readv326(local_bin)
        else:
            print(f"!! unknown version {ver} in {f}")
            return

        # ---- TIMESTAMP CONVERSION ----
        gps_epoch = datetime(1980, 1, 6)
        if 'week' in df.columns and 'towe' in df.columns:
            df['datetime'] = gps_epoch + pd.to_timedelta(df['week'].astype(int), unit='W') \
                             + pd.to_timedelta(df['towe'].astype(float), unit='s')
            df.drop(columns=['week', 'towe'], inplace=True, errors='ignore')

        # ---- WRITE PARQUET LOCALLY ----
        print(f"[WRITE LOCAL] Writing parquet -> {local_pq}")
        df.to_parquet(local_pq, compression='brotli',compression_level=9,index=False)

        # ---- MOVE FINAL FILE BACK TO /titan ----
        final_dir = os.path.dirname(pq_location)
        os.makedirs(final_dir, exist_ok=True)  # ensure destination exists
        print(f"[MOVE] Moving {local_pq} -> {pq_location}")
        shutil.move(local_pq, pq_location)

        print(f"[DONE] {pq_location}")

    except Exception as e:
        print(f"[ERROR] {f}: {e}")

    finally:
        # ---- CLEANUP LOCAL TMP ----
        try:
            shutil.rmtree(local_tmpdir)
            print(f"[CLEAN] Removed local temp dir {local_tmpdir}")
        except Exception as e:
            print(f"[CLEAN FAIL] Could not remove local temp dir {local_tmpdir}: {e}")
if __name__ == "__main__": # flist must be defined above 
    n_workers = 4 # or manually set e.g. 8 w
    with Pool(n_workers) as p: 
        p.map(process_one, flist)