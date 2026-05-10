import pandas as pd
import numpy as np

# ── Load combined training data ────────────────────────────────────────────────
df = pd.read_csv(r"C:\Users\User\Documents\Bachelor-Thesis\result spreadsheets\combined_features2.csv")

print(f"Original shape: {df.shape}")

X = df.drop(columns=["filename", "label"])
meta = df[["filename", "label"]]

# ── Replace outliers iteratively until none remain ────────────────────────────
def replace_outliers_iterative(df_features, threshold=3, max_passes=3):
    df_clean = df_features.copy()
    
    for pass_num in range(1, max_passes + 1):
        outliers_replaced = 0
        
        for col in df_clean.columns:
            mean = df_clean[col].mean()
            std  = df_clean[col].std()
            
            if std == 0:
                continue
            
            z_scores = np.abs((df_clean[col] - mean) / std)
            outlier_mask = z_scores > threshold
            count = outlier_mask.sum()
            
            if count > 0:
                clean_mean = df_clean.loc[~outlier_mask, col].mean()
                df_clean.loc[outlier_mask, col] = clean_mean
                outliers_replaced += count
        
        print(f"Pass {pass_num}: {outliers_replaced} outlier values replaced")
        
        if outliers_replaced == 0:
            print(f"Data is clean after {pass_num} passes!")
            break
    
    return df_clean

X_clean = replace_outliers_iterative(X)

# ── Recombine and save ─────────────────────────────────────────────────────────
df_clean = pd.concat([meta.reset_index(drop=True), X_clean.reset_index(drop=True)], axis=1)

print(f"\nFinal shape: {df_clean.shape}")
print(f"mfcc4 max after cleaning: {df_clean['mfcc4_sma3_stddevNorm'].max():.4f}")

df_clean.to_csv(r"C:\Users\User\Documents\Bachelor-Thesis\result spreadsheets\combined_features2_cleaner.csv", index=False)
print("Saved as combined_features2_cleaner.csv")