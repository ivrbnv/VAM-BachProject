import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

# Load combined data
df = pd.read_csv(r"C:\Users\User\Documents\Bachelor-Thesis\result spreadsheets\combined_features.csv")

# Drop filename, separate features and label
X = df.drop(columns=["filename", "label"])
y = df["label"]

# Split FIRST, then standardize
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_val, X_test, y_val, y_test     = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Fit scaler on training data only, then transform all splits
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# Save scaler for later use
joblib.dump(scaler, r"C:\Users\User\Documents\Bachelor-Thesis\python\scaler.pkl")

# Sanity check
print("Train size:", X_train.shape)
print("Val size:  ", X_val.shape)
print("Test size: ", X_test.shape)
print("\nTrain label balance:")
print(y_train.value_counts())

'''np.save(r"C:\Users\User\Documents\Bachelor-Thesis\python\X_train.npy", X_train)
np.save(r"C:\Users\User\Documents\Bachelor-Thesis\python\X_val.npy", X_val)
np.save(r"C:\Users\User\Documents\Bachelor-Thesis\python\X_test.npy", X_test)
y_train.to_csv(r"C:\Users\User\Documents\Bachelor-Thesis\python\y_train.csv", index=False)
y_val.to_csv(r"C:\Users\User\Documents\Bachelor-Thesis\python\y_val.csv", index=False)
y_test.to_csv(r"C:\Users\User\Documents\Bachelor-Thesis\python\y_test.csv", index=False)'''