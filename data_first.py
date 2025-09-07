import pandas as pd

# Load dataset
file_path = "videos_MHMisinfo_Full.csv"
df = pd.read_csv(file_path)

# Handle Missing Values
df['video_description'] = df['video_description'].fillna("")
df['audio_transcript'] = df['audio_transcript'].fillna("")

# Standardize Formats
text_columns = ['video_title', 'video_description', 'audio_transcript', 'platform']
for col in text_columns:
    df[col] = df[col].astype(str).str.strip()

# Normalize platform names
df['platform'] = df['platform'].str.lower()

# Encode Labels
label_map = {-1: "misinformation", 0: "neutral/reliable"}
df['label'] = df['label'].map(label_map)


# Remove Exact Duplicates
df_cleaned = df.drop_duplicates()

# Save cleaned dataset
df_cleaned.to_csv("videos_MHMisinfo_Full_Cleaned.csv", index=False)

print("Cleaning complete. Cleaned dataset saved as videos_MHMisinfo_Full_Cleaned.csv")
