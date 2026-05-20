import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import joblib

# ── Preprocessing ──────────────────────────────────────────────────────────────
df = pd.read_csv(r"C:\Users\User\Documents\Bachelor-Thesis\result spreadsheets\combined_features2_cleaner.csv")

X = df.drop(columns=["filename", "label"])
y = df["label"]

# Only need train/val split now — test set is cherry picked
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)

# ── Load cherry picked test set ────────────────────────────────────────────────
test_df = pd.read_csv(r"C:\Users\User\Documents\Bachelor-Thesis\result spreadsheets\combined_features_testing_clean.csv")

X_test = test_df.drop(columns=["filename", "label"])
y_test = test_df["label"]
X_test = scaler.transform(X_test)

# ── Models ─────────────────────────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest":       RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
    "SVM":                 SVC(kernel="rbf", class_weight="balanced", random_state=42),
}

# ── Train, evaluate on val, then test ─────────────────────────────────────────
for name, model in models.items():
    model.fit(X_train, y_train)

    # Validation results
    preds_val = model.predict(X_val)
    print(f"\n{'='*50}")
    print(f"  {name} — VALIDATION")
    print(f"{'='*50}")
    print(classification_report(y_val, preds_val, target_names=["Human", "AI"]))

    # Cherry picked test set results
    preds_test = model.predict(X_test)
    print(f"\n{'='*50}")
    print(f"  {name} — CHERRY PICKED TEST SET")
    print(f"{'='*50}")
    print(classification_report(y_test, preds_test, target_names=["Human", "AI"]))

    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=["Human", "AI"])
    plt.title(f"Confusion Matrix — {name} (Cherry Picked Test)")
    plt.tight_layout()
    plt.savefig(f"cm_{name.replace(' ', '_')}_cherrypicked.png")
    plt.show()

joblib.dump(scaler, r"C:\Users\User\Documents\Bachelor-Thesis\python\scaler.pkl")