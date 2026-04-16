import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib

# ── Preprocessing ──────────────────────────────────────────────────────────────
df = pd.read_csv(r"C:\Users\User\Documents\Bachelor-Thesis\result spreadsheets\combined_features.csv")

X = df.drop(columns=["filename", "label"])
y = df["label"]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_val, X_test, y_val, y_test     = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# ── Grid Search ────────────────────────────────────────────────────────────────
param_grid = {
    "n_estimators":      [100, 200, 300],
    "max_depth":         [3, 5, 7],
    "learning_rate":     [0.05, 0.1, 0.2],
    "min_samples_split": [2, 5],
}

grid = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)

print("\nBest parameters:", grid.best_params_)
print("Best cross-val F1:", round(grid.best_score_, 4))

# ── Evaluate best model on validation set ─────────────────────────────────────
best_model = grid.best_estimator_
preds = best_model.predict(X_val)

print("\n", classification_report(y_val, preds, target_names=["Human", "AI"]))

ConfusionMatrixDisplay.from_estimator(best_model, X_val, y_val, display_labels=["Human", "AI"])
plt.title("Confusion Matrix — Tuned Gradient Boosting")
plt.tight_layout()
plt.savefig("cm_tuned_gradient_boosting.png")
plt.show()

# ── Save best model ────────────────────────────────────────────────────────────
joblib.dump(best_model, r"C:\Users\User\Documents\Bachelor-Thesis\python\best_model.pkl")
joblib.dump(scaler, r"C:\Users\User\Documents\Bachelor-Thesis\python\scaler.pkl")

# Final honest evaluation
preds_test = best_model.predict(X_test)

print(classification_report(y_test, preds_test, target_names=["Human", "AI"]))

ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, display_labels=["Human", "AI"])
plt.title("Final Test Set — Tuned Gradient Boosting")
plt.tight_layout()
plt.savefig("cm_final_test.png")
plt.show()