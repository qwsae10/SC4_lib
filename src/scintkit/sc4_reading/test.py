import scintkit as sk
import pandas as pd 

file = '/Users/dal674840/Downloads/scintpi3_20241011_0004_96.7572W_32.9920N_v326f_lvl0.pq'

df = pd.read_parquet(file)

sk.pipelines.auto.process(file)