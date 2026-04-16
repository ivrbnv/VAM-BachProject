import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib

# ── Preprocessing ──────────────────────────────────────────────────────────────
df = pd.read_csv(r"C:\Users\User\Documents\Bachelor-Thesis\result spreadsheets\combined_features.csv")

X = df.drop(columns=["filename", "label"])
y = df["label"]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_val, X_test, y_val, y_test     = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

'scaler  = StandardScaler()'
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# ── Grid Search ────────────────────────────────────────────────────────────────
param_grid = {
    "C":      [0.1, 1, 10, 100],
    "gamma":  ["scale", "auto", 0.001, 0.01],
    "kernel": ["rbf", "sigmoid"],
}

grid = GridSearchCV(
    SVC(class_weight="balanced", random_state=42),
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)

print("\nBest parameters:", grid.best_params_)
print("Best cross-val F1:", round(grid.best_score_, 4))

# ── Evaluate on validation set ─────────────────────────────────────────────────
best_model = grid.best_estimator_
preds = best_model.predict(X_val)

print("\n", classification_report(y_val, preds, target_names=["Human", "AI"]))

ConfusionMatrixDisplay.from_estimator(best_model, X_val, y_val, display_labels=["Human", "AI"])
plt.title("Confusion Matrix — Tuned SVM")
plt.tight_layout()
plt.savefig("cm_tuned_svm.png")
plt.show()

# ── Final test set evaluation ──────────────────────────────────────────────────
preds_test = best_model.predict(X_test)

print("\nFinal Test Set Results:")
print(classification_report(y_test, preds_test, target_names=["Human", "AI"]))

ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, display_labels=["Human", "AI"])
plt.title("Final Test Set — Tuned SVM")
plt.tight_layout()
plt.savefig("cm_tuned_svm_test.png")
plt.show()

joblib.dump(best_model, r"C:\Users\User\Documents\Bachelor-Thesis\python\best_svm_model.pkl")

# ── Find misclassified AI samples ──────────────────────────────────────────────
test_indices = y_test.index
test_filenames = df.loc[test_indices, "filename"]

results = pd.DataFrame({
    "filename":  test_filenames.values,
    "actual":    y_test.values,
    "predicted": preds_test
})

ai_results = results[results["actual"] == 1]
print("\nAI samples in test set:")
print(ai_results.to_string(index=False))

# ── Feature comparison: missed vs caught AI samples ────────────────────────────
missed_files  = results[(results["actual"] == 1) & (results["predicted"] == 0)]["filename"].values
caught_files  = results[(results["actual"] == 1) & (results["predicted"] == 1)]["filename"].values

missed_features = df[df["filename"].isin(missed_files)].drop(columns=["filename", "label"])
caught_features = df[df["filename"].isin(caught_files)].drop(columns=["filename", "label"])

comparison = pd.DataFrame({
    "missed_mean": missed_features.mean(),
    "caught_mean": caught_features.mean(),
})

comparison["difference"] = (comparison["missed_mean"] - comparison["caught_mean"]).abs()
comparison = comparison.sort_values("difference", ascending=False)

print("\nTop 15 features where missed AI differs most from caught AI:")
print(comparison.head(15).to_string())

# Plot top 10
top_diff_features = comparison.head(10).index.tolist()

for feat in top_diff_features:
    plt.figure(figsize=(7, 3))
    sns.kdeplot(data=df[df["label"]==1], x=feat, fill=True, label="All AI")
    plt.axvline(df[df["filename"].isin(missed_files)][feat].mean(), 
                color="red", linestyle="--", label="Missed mean")
    plt.axvline(df[df["filename"].isin(caught_files)][feat].mean(), 
                color="green", linestyle="--", label="Caught mean")
    plt.title(feat)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"missed_vs_caught_{feat[:30]}.png")
    plt.show()

print(df[df["filename"].isin(missed_files)][["filename", "mfcc4_sma3_stddevNorm"]])