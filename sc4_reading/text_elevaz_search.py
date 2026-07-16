import pandas as pd

df = pd.read_csv(
    "/home/dal674840/scratch/txt_files/mx01316p15.25__SBF_Meas3Ranges.txt",
    header=None
)

print(df.head())
print(df.columns)
print(df.shape)