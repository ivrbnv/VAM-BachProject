import pandas as pd
#we use pandas to take the csv files
# Load all three files
ai1    = pd.read_csv(r"C:\Users\User\Documents\Bachelor-Thesis\result spreadsheets\ai_results2.csv")
ai2    = pd.read_csv(r"C:\Users\User\Documents\Bachelor-Thesis\result spreadsheets\ai_mp4_result2.csv")
ai3 = pd.read_csv(r"C:\Users\User\Documents\Bachelor-Thesis\result spreadsheets\ai_resultsDF.csv")
human  = pd.read_csv(r"C:\Users\User\Documents\Bachelor-Thesis\result spreadsheets\human_archive_results.csv")

# Combine the two AI files, then add labels
ai_df           = pd.concat([ai1, ai2, ai3], ignore_index=True)
ai_df["label"]  = 1  # AI = 1

human_df           = human.copy()
human_df["label"]  = 0  # Human = 0

# Unite everything into one dataframe
df = pd.concat([human_df, ai_df], ignore_index=True)

# Sanity check
print(df.shape)
print(df["label"].value_counts())

# Save
df.to_csv("combined_features2.csv", index=False)