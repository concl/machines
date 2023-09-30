
import pandas as pd

data = pd.read_parquet("machining/0000.parquet")

print(data["code"][0])

print(data.head())

