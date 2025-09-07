import pandas as pd
import numpy as np
import re

# Load dataset
file_path = "videos_MHMisinfo_Full.csv"
df = pd.read_csv(file_path)

# ----------------------
# 1. Handle Missing & Inconsistent Data
# ----------------------
df['video_description'] = df['video_description'].fillna("no_description")
df['audio_transcript'] = df['audio_transcript'].fillna("no_transcript")

# Drop rows where all text fields are missing/empty
df = df[~((df['video_title'].str.strip() == "") &
          (df['video_description'].str.strip() == "no_description") &
          (df['audio_transcript'].str.strip() == "no_transcript"))]

# ----------------------
# 2. Normalize Text Data
# ----------------------
def clean_text(text):
    text = str(text).lower()                     # lowercase
    text = re.sub(r"http\S+|www.\S+", "", text)  # remove URLs
    text = re.sub(r"[^a-z0-9\s#]", " ", text)    # keep alphanumeric + hashtags
    text = re.sub(r"\s+", " ", text).strip()     # remove extra spaces
    return text

for col in ['video_title', 'video_description', 'audio_transcript']:
    df[col] = df[col].apply(clean_text)

# Standardize platform names
df['platform'] = df['platform'].str.strip().str.lower()

# ----------------------
# 3. Encode Labels (make them human-readable)
# ----------------------
label_map = {-1: "misinformation", 0: "reliable"}
df['label'] = df['label'].map(label_map)

# ----------------------
# 4. Handle Numeric Outliers (log transform for skewed data)
# ----------------------
for col in ['video_view_count', 'video_like_count', 'video_comment_count']:
    df[f'{col}_log'] = np.log1p(df[col])  # log(1+x) to avoid log(0)

# ----------------------
# 5. Feature Engineering
# ----------------------
df['title_length'] = df['video_title'].apply(lambda x: len(x.split()))
df['desc_length'] = df['video_description'].apply(lambda x: len(x.split()))
df['transcript_length'] = df['audio_transcript'].apply(lambda x: len(x.split()))

# Engagement metrics (avoid division by zero)
df['like_ratio'] = df['video_like_count'] / df['video_view_count'].replace(0, np.nan)
df['comment_ratio'] = df['video_comment_count'] / df['video_view_count'].replace(0, np.nan)
df['like_ratio'] = df['like_ratio'].fillna(0)
df['comment_ratio'] = df['comment_ratio'].fillna(0)

# ----------------------
# 6. Ensure Unique Identifiers
# ----------------------
df = df.drop_duplicates(subset=['video_id'])

# ----------------------
# 7. Save Processed Dataset
# ----------------------
processed_path = "videos_MHMisinfo_Prepared.csv"
df.to_csv(processed_path, index=False)

print(f"✅ Cleaning complete. Cleaned dataset saved as {processed_path}")
print(df.head())
