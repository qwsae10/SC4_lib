
import pandas as pd
import numpy as np


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
    return dfin["cons"].map(constellation_map) + dfin["svid"].astype(int).astype(str).str.zfill(2) 
                                                                                               
def temp_formating(df):
    df= df.copy()

    gnssdic_loop = {0: 'GPS', 1: 'SBS', 2: 'GAL', 3: 'BDS', 6: 'GLO'}
    #check if cons is numeric
    s = pd.to_numeric(df['cons'], errors='coerce')

    if s.notna().all():
        df['cons'] = s.map(gnssdic_loop)
    df = df[~((df['cons'] == 'GLO') & (df['svid'] == 255))].copy()

    df=df.reset_index()
    df['minbin'] = df['datetime'].dt.floor('1min')
    df['prn']=make_prn(df)
    df=add_sigs(df)
    return df


def add_sigs(df):
    df = df.copy()

    mapping = {
    # sig1
    'GPS_L1CA': (1575.42, 'Sig1'), 'GLO_L1CA': (1602, 'Sig1'),
    'GEO_L1': (1575.42, 'Sig1'), 'QZS_L1CA': (1575.42, 'Sig1'),'GAL_L1BC': (1575.42, 'Sig1'),
    'GAL_E1': (1575.42, 'Sig1'), 'BDS_B1I': (1561.098, 'Sig1'),
    'IRNSS_L5': (1176.45, 'Sig1'),'GAL_L1BC': (1575.42, 'Sig1'),

    # sig2
    'GPS_L2C': (1227.60, 'Sig2'), 'GLO_L2C': (1246.60, 'Sig2'),'GLO_L2CA': (1246.60, 'Sig2'),
    'QZS_L2C': (1227.60, 'Sig2'), 'GAL_E5a': (1176.45, 'Sig2'),
    'SBAS_L5': (1176.45, 'Sig2'), 'BDS_B2I': (1207.14, 'Sig2'),
    'GEO_L5': (1176.45, 'Sig2'),

    # sig3
    'GPS_L5': (1176.45, 'Sig3'), 'QZS_L5': (1176.45, 'Sig3'),
    'GAL_E5b': (1207.14, 'Sig3'), 'BDS_B3I': (1268.52, 'Sig3'),
    
    #sig4
    'GPS_L2PY': (1227.60, 'Sig4'),
    'GPS_L1P': (1575.42, 'Sig4'),
    'GLO_L1P':(1602.00, 'Sig4'),
    'GLO_L2P':(1246.60, 'Sig4'),
    'GAL_E5':(1191.795, 'Sig4'),
    'GAL_E6BC':(1278.75, 'Sig4'),
    'GLO_L3':(1202.025, 'Sig4')
}
    
    hardcode_sig_dict={
        'GPS':{1:'GPS_L1CA',2:'GPS_L2C',3:'GPS_L5'},
        'GLO':{1:'GLO_L1CA',2:'GLO_L2C',3:'GLO_L3'},
        'GAL':{1:'GAL_L1BC',2:'GAL_E5a',3:'GAL_E5b'},
        'BDS':{1:'BDS_B1I',2:'BDS_B2I',3:'BDS_B3I'},
        'QZSS':{1:'QZS_L1CA',2:'QZS_L2C',3:'QZS_L5'}
        }

    if 'sig_1' not in df.columns:
        #scintpi3 doesn't have sig columns, but we can infer them from cons and svid. 
        #hardcoded for now, but could be made more flexible if needed
        df['sig_1'] = df.apply(lambda row: hardcode_sig_dict.get(row['cons'], {}).get(1, None), axis=1)
        df['sig_2'] = df.apply(lambda row: hardcode_sig_dict.get(row['cons'], {}).get(2, None), axis=1)

    df['freq_1'] = df['sig_1'].map(lambda x: mapping.get(x, (None, None))[0])
    df['freq_2'] = df['sig_2'].map(lambda x: mapping.get(x, (None, None))[0])

    return df

def make_1min(df,method='first'):

    """
    Resample the dataframe to 1 minute intervals, grouping by 'minbin' and 'prn'.
    Default method is 'first', but can be changed to any valid pandas aggregation method (e.g., 'mean', 'max', 'min').
    """

    df = df.copy()
    df=df.groupby(['minbin', 'prn']).agg(method).reset_index()
    return df