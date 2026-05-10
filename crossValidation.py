import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib

# ── Load training data ─────────────────────────────────────────────────────────
train = pd.read_csv(r"C:\Users\User\Documents\Bachelor-Thesis\result spreadsheets\combined_features2.csv")

X = train.drop(columns=["filename", "label"])
y = train["label"]

# ── Load fixed test set ────────────────────────────────────────────────────────
test = pd.read_csv(r"C:\Users\User\Documents\Bachelor-Thesis\result spreadsheets\combined_features_testing.csv")

X_test = test.drop(columns=["filename", "label"])
y_test = test["label"]

# ── Define models ──────────────────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, class_weight="balanced"),
    "Random Forest":       RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
    "SVM":                 SVC(kernel="rbf", class_weight="balanced", random_state=42),
}

# ── Stratified K-Fold Cross Validation ────────────────────────────────────────
skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)

print("="*60)
print("CROSS VALIDATION RESULTS (8-fold)")
print("="*60)

results = {}

for name, model in models.items():
    # Pipeline: scale then train inside each fold
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]

        # Scale inside the fold to prevent leakage
        scaler = StandardScaler()
        X_fold_train = scaler.fit_transform(X_fold_train)
        X_fold_val   = scaler.transform(X_fold_val)

        model.fit(X_fold_train, y_fold_train)
        preds = model.predict(X_fold_val)

        # Get F1 for AI class only
        report = classification_report(y_fold_val, preds, output_dict=True, zero_division=0)
        ai_f1 = report.get("1", {}).get("f1-score", 0)
        fold_scores.append(ai_f1)

    mean_f1 = np.mean(fold_scores)
    std_f1  = np.std(fold_scores)
    results[name] = {"mean_f1": mean_f1, "std_f1": std_f1}

    print(f"\n{name}")
    print(f"  AI F1 per fold: {[round(s, 2) for s in fold_scores]}")
    print(f"  Mean F1: {round(mean_f1, 3)} ± {round(std_f1, 3)}")

# ── Final evaluation on fixed test set ────────────────────────────────────────
print("\n" + "="*60)
print("FINAL TEST SET RESULTS")
print("="*60)

# Refit each model on full training data, scale properly
scaler_final = StandardScaler()
X_train_scaled = scaler_final.fit_transform(X)
X_test_scaled  = scaler_final.transform(X_test)

for name, model in models.items():
    model.fit(X_train_scaled, y)
    preds_test = model.predict(X_test_scaled)

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(classification_report(y_test, preds_test, target_names=["Human", "AI"]))

    ConfusionMatrixDisplay.from_estimator(model, X_test_scaled, y_test, display_labels=["Human", "AI"])
    plt.title(f"Final Test Set — {name}")
    plt.tight_layout()
    plt.savefig(f"cm_final_{name.replace(' ', '_')}.png")
    plt.show()

joblib.dump(scaler_final, r"C:\Users\User\Documents\Bachelor-Thesis\python\scaler_cross_validation_cleaned.pkl")