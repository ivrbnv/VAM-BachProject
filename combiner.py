import pandas as pd

# Load all three files
ai1    = pd.read_csv(r"C:\Users\User\Documents\Bachelor-Thesis\result spreadsheets\ai_results.csv")
ai2    = pd.read_csv(r"C:\Users\User\Documents\Bachelor-Thesis\result spreadsheets\ai_mp4_results.csv")
human  = pd.read_csv(r"C:\Users\User\Documents\Bachelor-Thesis\result spreadsheets\human_archive_results.csv")

# Combine the two AI files, then add labels
ai_df           = pd.concat([ai1, ai2], ignore_index=True)
ai_df["label"]  = 1  # AI = 1

human_df           = human.copy()
human_df["label"]  = 0  # Human = 0

# Unite everything into one dataframe
df = pd.concat([human_df, ai_df], ignore_index=True)

# Sanity check
print(df.shape)
print(df["label"].value_counts())

# Save
df.to_csv("combined_features.csv", index=False)