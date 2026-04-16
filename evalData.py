import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\User\Documents\Bachelor-Thesis\result spreadsheets\combined_features.csv")

# Basic info
print("Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())

# Class balance
df["label"].value_counts().plot(kind="bar", color=["steelblue", "tomato"])
plt.title("Class Balance: Human vs AI")
plt.xticks([0, 1], ["Human", "AI"], rotation=0)
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("class_balance.png")
plt.show()

# Summary statistics per class
X = df.drop(columns=["filename", "label"])
print(df.groupby("label")[X.columns].mean().T)

# Find the most promising features by relative difference between classes
means = df.groupby("label")[X.columns].mean()
diff = ((means.loc[1] - means.loc[0]) / means.loc[0]).abs().sort_values(ascending=False)

print("Top 15 most different features between Human and AI:")
print(diff.head(15))

# Plot distributions for top 10
'''top_features = diff.head(10).index.tolist()

for feat in top_features:
    plt.figure(figsize=(7, 3))
    sns.kdeplot(data=df, x=feat, hue="label", fill=True, common_norm=False)
    plt.title(feat)
    plt.tight_layout()
    plt.savefig(f"dist_{feat[:30]}.png")
    plt.show()
'''

plt.figure(figsize=(16, 14))
sns.heatmap(
    X.corr(),
    cmap="coolwarm",
    center=0,
    linewidths=0.2,
    xticklabels=False,  # too many to display cleanly
    yticklabels=False
)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()