
import kagglehub
import pandas as pd
# Download latest version
path = r"C:\Users\keenu\Downloads\archive\final_dataset.csv"
data = pd.read_csv(path)
print(data.columns)
