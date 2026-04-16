import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\User\Documents\Bachelor-Thesis\result spreadsheets\combined_features.csv")

X = df.drop(columns=["filename", "label"])

# Calculate z-scores for every value
z_scores = np.abs((X - X.mean()) / X.std())

# Find any row/column combination where z-score exceeds 3
threshold = 3
outliers = []

for col in X.columns:
    extreme_rows = df[z_scores[col] > threshold][["filename", "label", col]]
    for _, row in extreme_rows.iterrows():
        outliers.append({
            "filename": row["filename"],
            "label":    row["label"],
            "feature":  col,
            "value":    row[col],
            "z_score":  z_scores.loc[row.name, col]
        })

outlier_df = pd.DataFrame(outliers).sort_values("z_score", ascending=False)
print(f"Total extreme values found: {len(outlier_df)}")
print("\nTop 20 most extreme values:")
print(outlier_df.head(20).to_string(index=False))

# Summarise by filename - which files have the most outlier features
print("\nFiles with most outlier features:")
print(outlier_df.groupby(["filename", "label"])["feature"].count().sort_values(ascending=False).head(10))

print(f"\nTotal unique files with outliers: {outlier_df['filename'].nunique()}")
print(f"Total human files with outliers: {outlier_df[outlier_df['label']==0]['filename'].nunique()}")
print(f"Total AI files with outliers: {outlier_df[outlier_df['label']==1]['filename'].nunique()}")