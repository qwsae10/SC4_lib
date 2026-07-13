import pandas as pd

file = "mx01316a00.25__SBF_Meas3Ranges.txt"

cols = [
    "tow",
    "week",
    "prn",
    "signal",
    "antenna",
    "pseudorange",
    "carrier_phase",
    "unused",
    "cn0",
    "lock_time"
]

df = pd.read_csv(
    file,
    names=cols,
    header=None
)

print(df.head())
print(df.info())