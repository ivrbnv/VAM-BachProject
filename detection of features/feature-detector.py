import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ── Load data ──────────────────────────────────────────────────────────────────
train = pd.read_csv(r"C:\Users\User\Documents\Bachelor-Thesis\result spreadsheets\combined_features2_cleaner.csv")
test  = pd.read_csv(r"C:\Users\User\Documents\Bachelor-Thesis\result spreadsheets\combined_features_testing_clean.csv")

X       = train.drop(columns=["filename", "label"])
y       = train["label"]
X_test  = test.drop(columns=["filename", "label"])
y_test  = test["label"]

# ── Scale ──────────────────────────────────────────────────────────────────────
scaler  = StandardScaler()
X_scaled      = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# ── Approach 1: Random Forest Feature Importance ───────────────────────────────
print("="*60)
print("APPROACH 1 — RANDOM FOREST FEATURE IMPORTANCE")
print("="*60)

rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)
rf.fit(X_scaled, y)

importances = pd.Series(rf.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

print("\nTop 20 most important features:")
print(importances.head(20).round(4).to_string())

print("\nBottom 10 least important features:")
print(importances.tail(10).round(4).to_string())

# Plot
plt.figure(figsize=(12, 6))
importances.head(20).plot(kind="bar")
plt.title("Top 20 Feature Importances (Random Forest)")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importances.png")
print("\nSaved feature_importances.png")

# ── Approach 2: Performance vs Number of Features ─────────────────────────────
print("\n" + "="*60)
print("APPROACH 2 — PERFORMANCE VS NUMBER OF FEATURES")
print("="*60)

# Test SVM with decreasing number of features
feature_counts = [88, 70, 50, 30, 20, 15, 10, 5]
results = []

# Use top N features by Random Forest importance
for n in feature_counts:
    top_features = importances.head(n).index.tolist()
    
    X_subset      = X[top_features]
    X_test_subset = X_test[top_features]
    
    scaler_sub       = StandardScaler()
    X_subset_scaled      = scaler_sub.fit_transform(X_subset)
    X_test_subset_scaled = scaler_sub.transform(X_test_subset)
    
    # Cross val score
    skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
    svm = SVC(kernel="rbf", class_weight="balanced", random_state=42)
    cv_scores = cross_val_score(svm, X_subset_scaled, y, cv=skf, scoring="f1", n_jobs=-1)
    
    # Test set score
    svm.fit(X_subset_scaled, y)
    preds = svm.predict(X_test_subset_scaled)
    report = classification_report(y_test, preds, output_dict=True, target_names=["Human", "AI"])
    test_f1 = report["AI"]["f1-score"]
    
    results.append({
        "n_features": n,
        "cv_mean_f1": round(cv_scores.mean(), 3),
        "cv_std":     round(cv_scores.std(), 3),
        "test_f1":    round(test_f1, 3)
    })
    
    print(f"Features: {n:3d} | CV F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f} | Test F1: {test_f1:.3f}")

# Plot performance curve
results_df = pd.DataFrame(results)

plt.figure(figsize=(10, 5))
plt.plot(results_df["n_features"], results_df["cv_mean_f1"], marker="o", label="CV Mean F1", color="steelblue")
plt.plot(results_df["n_features"], results_df["test_f1"],    marker="s", label="Test F1",    color="tomato")
plt.fill_between(
    results_df["n_features"],
    results_df["cv_mean_f1"] - results_df["cv_std"],
    results_df["cv_mean_f1"] + results_df["cv_std"],
    alpha=0.2, color="steelblue"
)
plt.xlabel("Number of Features")
plt.ylabel("F1 Score (AI class)")
plt.title("SVM Performance vs Number of Features")
plt.legend()
plt.gca().invert_xaxis()
plt.tight_layout()
plt.savefig("feature_selection_curve.png")
print("\nSaved feature_selection_curve.png")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(results_df.to_string(index=False))
best = results_df.loc[results_df["test_f1"].idxmax()]
print(f"\nBest result: {int(best['n_features'])} features with test F1 of {best['test_f1']}")