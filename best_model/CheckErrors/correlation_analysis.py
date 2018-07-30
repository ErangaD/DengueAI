import pandas as pd
import numpy as np

df = pd.read_csv("dengue_features_train.csv", index_col=[0, 1, 2])
df2=pd.read_csv("dengue_labels_train.csv", index_col=[0, 1, 2])
print(df['station_precip_mm'].corr(df2['total_cases']))