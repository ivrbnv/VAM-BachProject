import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\User\Documents\Bachelor-Thesis\result spreadsheets\combined_features_cleaner.csv")
X = df.drop(columns=["filename", "label"])
z_scores = np.abs((X - X.mean()) / X.std())
remaining = (z_scores > 3).sum().sum()
print(f"Remaining outlier values after 3 passes: {remaining}")
print(f"mfcc4 max: {df['mfcc4_sma3_stddevNorm'].max():.4f}")
print(f"mfcc4 mean: {df['mfcc4_sma3_stddevNorm'].mean():.4f}")