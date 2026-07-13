import pandas as pd
from pathlib import Path
import numpy as np
import os
from pathlib import Path

file = Path("/home/dal674840/scratch/txt_files/mx01316a00.25__measurements.txt")

with open(file, "r", errors="ignore") as f:
    lines = f.readlines()

print(lines[:10])      # First 10 lines
print(f"Total lines: {len(lines)}")