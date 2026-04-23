import pandas as pd





cols=['Datetime', 'SVID', 'Elevation', 'Azimuth', 'SNR', 'Phase']


gnssdic = {0: "GPS", 1: "SBS", 2: "GAL", 3: "BDS", 6: "GLO"}


def make_prn(dfin):
    constellation_map = {
        "GPS": "G",
        "BDS": "C",
        "GAL": "E",
        "GLO": "R",
        "QZSS": "J",
        "IRNSS": "I",
        "SBAS": "S",
        "SBS": "S",
    }
    return dfin["SIG"].map(constellation_map) + dfin["SVID"].astype(int).astype(str).str.zfill(2 