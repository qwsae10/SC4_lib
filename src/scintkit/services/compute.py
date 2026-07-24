#%%
import pandas as pd
import numpy as np

from scintkit.services.phase_detrend import process_phases,repair_discontinuities_pos,detect_sampling_rate
from scintkit.preprocessing.format import temp_formating


import numpy as np
import pdb

def carrier_phase_tec(phi1_cyc, phi2_cyc, f1_hz, f2_hz):
    c = 299792458  # m/s

    lambda1 = c / f1_hz
    lambda2 = c / f2_hz

    phi1_m = phi1_cyc * lambda1 
    phi2_m = phi2_cyc * lambda2 

    tec_factor = (f1_hz**2 * f2_hz**2) / (40.3 * (f1_hz**2 - f2_hz**2))

    return tec_factor * (phi1_m - phi2_m)/1e16


def pseudorange_tec(P1_m, P2_m, f1_hz, f2_hz):
    tec_factor = (f1_hz**2 * f2_hz**2) / (40.3 * (f1_hz**2 - f2_hz**2))

    return tec_factor * (P2_m - P1_m)/1e16


def add_tec_columns(df, pair="13", fs=None, max_gap="5min"):
    """Add carrier-phase and pseudorange TEC for a frequency pair.

    Each PRN is split into continuous time segments. A time gap strictly
    greater than ``max_gap`` starts a new segment. Carrier and pseudorange TEC
    are repaired independently within each segment, then the carrier TEC is
    shifted so that its segment median matches the pseudorange TEC median at
    common valid epochs. Carrier TEC is left missing when a segment has no
    pseudorange overlap and therefore cannot be leveled.
    """
    df = df.reset_index(drop=True).copy()
    gap_threshold = pd.to_timedelta(max_gap)

def add_tec_columns(df, pair="13", fs=None):
    df = df.reset_index(drop=True).copy() 



    def _per_prn(key, g):
        # Phase repair and gap detection both depend on chronological order.
        # Keep the original index so the caller's row order can be restored.
        if "datetime" in g.columns:
            g = g.sort_values("datetime", kind="stable").copy()
            time = pd.to_datetime(g["datetime"], errors="coerce")
        else:
            g = g.copy()
            time = None

        N1 = pair[0]
        N2 = pair[1]

        phi1 = g[f"cph{N1}"]
        phi2 = g[f"cph{N2}"]
        # A zero pseudorange is the receiver's missing-value sentinel, not a
        # physical range. Treat it as missing before computing TEC.
        rng1 = g[f"rng{N1}"].replace(0, np.nan)
        rng2 = g[f"rng{N2}"].replace(0, np.nan)
        f1_hz = g[f"freq_{N1}"] * 1e6
        f2_hz = g[f"freq_{N2}"] * 1e6

        carrier_valid = (
            phi1.notna()
            & phi2.notna()
            & f1_hz.notna()
            & f2_hz.notna()
            & f1_hz.ne(f2_hz)
        )

        pseudo_valid = (
            rng1.notna()
            & rng2.notna()
            & f1_hz.notna()
            & f2_hz.notna()
            & f1_hz.ne(f2_hz)
        )

        if time is not None:
            new_segment = time.diff().gt(gap_threshold) | time.isna()

            # A PRN can remain in the table while one carrier-phase channel is
            # missing. Split when valid carrier data resume after a long
            # outage, even if the dataframe still has intervening timestamps.
            previous_carrier_time = time.where(carrier_valid).ffill().shift()
            carrier_gap = (
                carrier_valid
                & time.sub(previous_carrier_time).gt(gap_threshold)
            )
            new_segment |= carrier_gap
            new_segment.iloc[0] = False
            segment_id = new_segment.cumsum()
        else:
            segment_id = pd.Series(0, index=g.index)

        carrier_raw = carrier_phase_tec(
            phi1_cyc=phi1.where(carrier_valid),
            phi2_cyc=phi2.where(carrier_valid),
            f1_hz=f1_hz.where(carrier_valid),
            f2_hz=f2_hz.where(carrier_valid),
        )
        pseudo_raw = pseudorange_tec(
            P1_m=rng1.where(pseudo_valid),
            P2_m=rng2.where(pseudo_valid),
            f1_hz=f1_hz.where(pseudo_valid),
            f2_hz=f2_hz.where(pseudo_valid),
        )

        carrier = pd.Series(np.nan, index=g.index, dtype=float)
        pseudo = pd.Series(np.nan, index=g.index, dtype=float)

        segment_groups = segment_id.groupby(segment_id, sort=False).groups
        for segment_index in segment_groups.values():
            segment_index = pd.Index(segment_index)

            carrier_segment, _, _ = repair_discontinuities_pos(
                carrier_raw.loc[segment_index],
                fs=fs,
                threshold=1,
                svid=key,
                verbose=True,
            )
            pseudo_segment, _, _ = repair_discontinuities_pos(
                pseudo_raw.loc[segment_index],
                fs=fs,
                threshold=1,
                svid=key,
                verbose=False,

        # if carrier inputs invalid
        if not carrier_valid.any():
            carrier = np.full(len(g), np.nan)
            n_slip_carrier = 0
        else:
            carrier = carrier_phase_tec(
                phi1_cyc=phi1.where(carrier_valid),
                phi2_cyc=phi2.where(carrier_valid),
                f1_hz=f1_hz.where(carrier_valid),
                f2_hz=f2_hz.where(carrier_valid),
            )

            carrier, _, n_slip_carrier = repair_discontinuities_pos(
                carrier, fs=fs, threshold=1, svid=key, verbose=True
            )

            carrier = carrier - np.nanmean(carrier)

        # if pseudorange inputs invalid
        pseudo_valid = (
            rng1.notna()
            & rng2.notna()
            & f1_hz.notna()
            & f2_hz.notna()
            & f1_hz.ne(f2_hz)
        )
        if not pseudo_valid.any():
            pseudo = np.full(len(g), np.nan)
            n_slip_pseudo = 0
        else:
            pseudo = pseudorange_tec(
                P1_m=rng1.where(pseudo_valid),
                P2_m=rng2.where(pseudo_valid),
                f1_hz=f1_hz.where(pseudo_valid),
                f2_hz=f2_hz.where(pseudo_valid),
            )

            pseudo, _, n_slip_pseudo = repair_discontinuities_pos(
                pseudo, fs=fs, threshold=1, svid=key, verbose=False
            )

            common_valid = carrier_segment.notna() & pseudo_segment.notna()
            if common_valid.any():
                carrier_median = carrier_segment.loc[common_valid].median()
                pseudo_median = pseudo_segment.loc[common_valid].median()
                carrier_segment = carrier_segment + (
                    pseudo_median - carrier_median
                )
            else:
                carrier_segment[:] = np.nan

            carrier.loc[segment_index] = carrier_segment.to_numpy()
            pseudo.loc[segment_index] = pseudo_segment.to_numpy()

        g[f"tec_cph{pair}"] = carrier
        g[f"tec_rng{pair}"] = pseudo

        return g
    out = pd.concat(
        [_per_prn(key, g) for key, g in df.groupby("prn", sort=False)]
    )

    return out.sort_index(kind="stable").reset_index(drop=True)

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


SIGMA_PHI_MAX_DROPPED_SAMPLES = 10
S4_MIN_SAMPLE_FRACTION = 0.8


def _add_quality_flags(products, fs):
    """Add binary, per-frequency quality flags to minute products."""
    if fs is None or not np.isfinite(fs) or fs <= 0:
        raise ValueError(
            "A positive sampling rate is required to compute quality flags."
        )

    expected_samples = fs * 60
    sigma_phi_min_samples = expected_samples - SIGMA_PHI_MAX_DROPPED_SAMPLES
    s4_min_samples = expected_samples * S4_MIN_SAMPLE_FRACTION
    is_glonass = products["prn"].astype(str).str.startswith("R")

    internal_columns = []
    for i in ("1", "2", "3"):
        phase_count_col = f"n_{i}"
        edge_gap_col = f"_sigma_phi_edge_gap_{i}"
        s4_count_col = f"_s4_sample_count_{i}"

        if phase_count_col in products.columns:
            if edge_gap_col in products.columns:
                has_edge_gap = products[edge_gap_col].astype(bool)
            else:
                # The phase product is not trustworthy if its edge/gap mask
                # was not propagated from phase detrending. Treat a missing
                # per-frequency mask as bad instead of silently assuming that
                # the channel contains no edge or gap contamination.
                has_edge_gap = pd.Series(True, index=products.index)
                has_edge_gap = pd.Series(False, index=products.index)

            sigma_phi_bad = (
                has_edge_gap
                | products[phase_count_col].lt(sigma_phi_min_samples)
                #| is_glonass #maybe add this back in later if we want to filter out GLONASS
            )
            products[f"sigma_phi_quality_flag_{i}"] = sigma_phi_bad.astype(
                np.int8
            )

        if s4_count_col in products.columns:
            products[f"s4_quality_flag_{i}"] = products[s4_count_col].lt(
                s4_min_samples
            ).astype(np.int8)

        internal_columns.extend([edge_gap_col, s4_count_col])

    return products.drop(columns=internal_columns, errors="ignore")


def add_products(df,verbose=False):

def add_products(df,verbose=False,fs=None):
    """
    This function takes a full-rate dataframe (fs=20 or 10 Hz) at and computes various products:
    - tec12 and tec13: differences between detrended phases to estimate TEC (WIP)
    - sigma_phi_1, sigma_phi_2, sigma_phi_3: standard deviation of detrended phases with clock noise removed, for each frequency
    - n_1, n_2, n_3: number of valid samples for each frequency
    - n_cycleslip_1, n_cycleslip_2, n_cycleslip_3: number of detected cycle slips for each phase
    - sigma_phi_quality_flag_1/2/3: binary sigma-phi quality flags; 0 is good and 1 marks an edge/gap, too many dropped samples, or GLONASS
    - s4_quality_flag_1/2/3: binary S4 quality flags; 0 is good and 1 marks fewer than 80% of the expected samples
    - s4_1, s4_2, s4_3: S4 index computed from SNR values for each frequency
    - s4_corrected_1, s4_corrected_2, s4_corrected_3: S4 index corrected for bias based on Van Dierendonck (1993) method
    The function groups the data by PRN and 1-minute bins to compute these products, and then merges the results back to the original dataframe in the same time bins.
    """

    if verbose:
        print("Ensuring format...")
    df=temp_formating(df)
    if verbose:
        print("Processing phases...")   
    df = process_phases(df)
    if not(fs):
        fs=detect_sampling_rate(df)
    
    if verbose:
        print("Computing TEC...")

    if f"cph1" in df.columns and f"cph2" in df.columns:
        df=add_tec_columns(df,fs=fs, pair="12")
    if f"cph1" in df.columns and f"cph3" in df.columns:
        df=add_tec_columns(df,fs=fs, pair="13")

    
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

        # Preserve the edge/gap result until it can be combined with the
        # sample-count and constellation checks below.
        if edgegap_col in df.columns:
            agg_dict[f"_sigma_phi_edge_gap_{i}"] = (
                edgegap_col,
                lambda x: int(x.fillna(False).astype(bool).any())
            )

        if snr_col in df.columns:
            agg_dict[f"s4_{i}"] = (snr_col, compute_s4)
            agg_dict[f"s4_corrected_{i}"] = (snr_col, compute_s4_corrected)
            agg_dict[f"_s4_sample_count_{i}"] = (
                snr_col,
                compute_n_samples,
            )
    if not agg_dict:
        return df

    products = (
        df.groupby(group_cols, sort=False)
        .agg(**agg_dict)
        .reset_index()
    )
    products = _add_quality_flags(products, fs=fs)
    if verbose:
        print("Merging products back to original dataframe...")
    df = df.merge(products, on=group_cols, how="left")

    return df

# %%
