import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\User\Documents\Bachelor-Thesis\result spreadsheets\combined_features.csv")

X = df.drop(columns=["filename", "label"])

# Calculate z-scores
z_scores = np.abs((X - X.mean()) / X.std())

# Find all outliers
outliers = []

for col in X.columns:
    extreme_rows = df[z_scores[col] > 3][["filename", "label", col]]
    for _, row in extreme_rows.iterrows():
        outliers.append({
            "filename": row["filename"],
            "label":    "AI" if row["label"] == 1 else "Human",
            "feature":  col,
            "value":    row[col],
            "z_score":  round(z_scores.loc[row.name, col], 2)
        })

outlier_df = pd.DataFrame(outliers).sort_values("z_score", ascending=False)

# Save full report
outlier_df.to_csv(r"C:\Users\User\Documents\Bachelor-Thesis\python\outlier_report.csv", index=False)

# Save summary by file
summary = outlier_df.groupby(["filename", "label"])["feature"].count().reset_index()
summary.columns = ["filename", "label", "outlier_feature_count"]
summary = summary.sort_values("outlier_feature_count", ascending=False)
summary.to_csv(r"C:\Users\User\Documents\Bachelor-Thesis\python\outlier_summary.csv", index=False)

print(f"Full report saved: {len(outlier_df)} outlier values across {outlier_df['filename'].nunique()} files")
print(f"\nTop 10 most problematic files:")
print(summary.head(10).to_string(index=False))