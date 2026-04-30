#%%
import pandas as pd
import numpy as np

from scintkit.services.phase_detrend import process_phases
from scintkit.preprocessing.format import temp_formating
def compute_tec(f1,f2):
    return f1 - f2

def compute_s4(snr):
    snr = snr.dropna()
    if len(snr) == 0:
        return np.nan

    lin_snr = 10 ** (snr / 10)
    mean = np.mean(lin_snr)
    std = np.std(lin_snr)

    return std / mean if mean > 0 else np.nan


def compute_s4_corrected(snr):
    snr = snr.dropna()
    if len(snr) == 0:
        return np.nan

    lin_snr = 10 ** (snr / 10)
    mean = np.mean(lin_snr)
    std = np.std(lin_snr)

    if mean <= 0:
        return np.nan

    s4 = std / mean
    s4_correction = np.sqrt(100 / mean * (1 + 500 / (19 * mean)))
    val = s4**2 - s4_correction**2
    return np.sqrt(val) if val > 0 else 0


def compute_n_cycleslips(cycleslips):
    return int(cycleslips.fillna(False).sum())


def compute_n_samples(col):
    return int(col.notna().sum())


def compute_sigma_phi(phase):
    phase = phase.dropna()
    return np.std(phase) if len(phase) > 0 else np.nan


def add_products(df,verbose=False):
    """
    This function takes a full-rate dataframe (fs=20 or 10 Hz) at and computes various products:
    - tec12 and tec13: differences between detrended phases to estimate TEC (WIP)
    - sigma_phi_1, sigma_phi_2, sigma_phi_3: standard deviation of detrended phases with clock noise removed, for each frequency
    - n_1, n_2, n_3: number of valid samples for each frequency
    - n_cycleslip_1, n_cycleslip_2, n_cycleslip_3: number of detected cycle slips for each phase
    - quality_1, quality_2, quality_3: binary flags indicating potential quality issues (0 means no issue, 1 or more means issue) 
    - s4_1, s4_2, s4_3: S4 index computed from SNR values for each frequency
    - s4_corrected_1, s4_corrected_2, s4_corrected_3: S4 index corrected for bias based on Van Dierendonck (1993) method
    The function groups the data by PRN and 1-minute bins to compute these products, and then merges the results back to the original dataframe in the same time bins.
    """

    df = df.copy()

    if verbose:
        print("Ensuring format...")
    df=temp_formating(df)
    if verbose:
        print("Processing phases...")   
    df = process_phases(df)

    if verbose:
        print("Computing TEC...")
    if ('detrended_cph1' in df.columns) and ('detrended_cph2' in df.columns):
        df['tec12'] = compute_tec(df['detrended_cph1'], df['detrended_cph2'])

    if ('detrended_cph1' in df.columns) and ('detrended_cph3' in df.columns):
        df['tec13'] = compute_tec(df['detrended_cph1'], df['detrended_cph3'])

    if verbose:
        print("Computing products...")

    group_cols = ["prn", "minbin"]
    agg_dict = {}
    for i in ("1", "2", "3"):


        detrended_noclk_col = f"detrended_noclk_cph{i}"
        cycleslip_col = f"cycleslips_cph{i}"
        edgegap_col = f"edgegap_mask_cph{i}"
        snr_col = f"snr{i}"

        if detrended_noclk_col in df.columns:
            agg_dict[f"sigma_phi_{i}"] = (detrended_noclk_col, compute_sigma_phi)
            agg_dict[f"n_{i}"] = (detrended_noclk_col, compute_n_samples)

        if cycleslip_col in df.columns:
            agg_dict[f"n_cycleslip_{i}"] = (cycleslip_col, compute_n_cycleslips)

        #if close to edge of pass, mark as potential quality issue
        if edgegap_col in df.columns:
            agg_dict[f"quality_{i}"] = (
                edgegap_col,
                lambda x: int(x.fillna(False).astype(bool).any())
            )

        if snr_col in df.columns:
            agg_dict[f"s4_{i}"] = (snr_col, compute_s4)
            agg_dict[f"s4_corrected_{i}"] = (snr_col, compute_s4_corrected)
    if not agg_dict:
        return df

    products = (
        df.groupby(group_cols, sort=False)
        .agg(**agg_dict)
        .reset_index()
    )
    if verbose:
        print("Merging products back to original dataframe...")
    df = df.merge(products, on=group_cols, how="left")

    return df
