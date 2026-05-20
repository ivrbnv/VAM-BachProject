import pandas as pd
import numpy as np

# ── Load test data and training statistics ─────────────────────────────────────
test = pd.read_csv(r"C:\Users\User\Documents\Bachelor-Thesis\result spreadsheets\combined_features_testing.csv")
clean_means = pd.read_csv(r"C:\Users\User\Documents\Bachelor-Thesis\python\train_clean_means.csv", index_col=0).squeeze()
clean_stds  = pd.read_csv(r"C:\Users\User\Documents\Bachelor-Thesis\python\train_clean_stds.csv", index_col=0).squeeze()

print(f"Original test shape: {test.shape}")

X_test = test.drop(columns=["filename", "label"])
meta   = test[["filename", "label"]]

# ── Replace outliers using TRAINING means and stds ────────────────────────────
def clean_test_with_train_stats(df_test, means, stds, threshold=3, max_passes=3):
    df_clean = df_test.copy()

    for pass_num in range(1, max_passes + 1):
        outliers_replaced = 0

        for col in df_clean.columns:
            if stds[col] == 0:
                continue

            z_scores = np.abs((df_clean[col] - means[col]) / stds[col])
            outlier_mask = z_scores > threshold
            count = outlier_mask.sum()

            if count > 0:
                df_clean.loc[outlier_mask, col] = means[col]
                outliers_replaced += count

        print(f"Pass {pass_num}: {outliers_replaced} outlier values replaced")

        if outliers_replaced == 0:
            print(f"Test data clean after {pass_num} passes!")
            break

    return df_clean

X_test_clean = clean_test_with_train_stats(X_test, clean_means, clean_stds)

# ── Recombine and save ─────────────────────────────────────────────────────────
test_clean = pd.concat([meta.reset_index(drop=True), X_test_clean.reset_index(drop=True)], axis=1)

print(f"\nFinal test shape: {test_clean.shape}")
test_clean.to_csv(r"C:\Users\User\Documents\Bachelor-Thesis\result spreadsheets\combined_features_testing_clean.csv", index=False)
print("Saved as combined_features_testing_clean.csv")