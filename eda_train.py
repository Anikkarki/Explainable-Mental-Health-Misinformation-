import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# =====================
# 1. Load Cleaned Dataset
# =====================
file_path = "videos_MHMisinfo_Prepared.csv"
df = pd.read_csv(file_path)

print("Dataset Loaded")
print("Shape:", df.shape)
print("\nClass Distribution:\n", df['label'].value_counts())

# =====================
# 2. Exploratory Data Analysis (EDA)
# =====================

# --- Missing Values
print("\nMissing Values:\n", df.isnull().sum())

# --- Label Distribution
plt.figure(figsize=(5,4))
sns.countplot(x='label', data=df, palette="Set2")
plt.title("Distribution of Labels (Misinformation vs Reliable)")
plt.show()

# --- Platform Distribution
plt.figure(figsize=(7,5))
sns.countplot(x='platform', data=df, order=df['platform'].value_counts().index, palette="Set3")
plt.title("Video Count by Platform")
plt.xticks(rotation=45)
plt.show()

# --- Distribution of Numeric Features
numeric_cols = ['video_view_count', 'video_like_count', 'video_comment_count',
                'video_view_count_log', 'video_like_count_log', 'video_comment_count_log',
                'title_length', 'desc_length', 'transcript_length', 'like_ratio', 'comment_ratio']

for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], bins=50, kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

# --- Compare Reliable vs Misinformation
plt.figure(figsize=(7,5))
sns.boxplot(x='label', y='video_view_count_log', data=df)
plt.title("Views (log scale) by Label")
plt.show()

plt.figure(figsize=(7,5))
sns.boxplot(x='label', y='like_ratio', data=df)
plt.title("Like Ratio by Label")
plt.show()

plt.figure(figsize=(7,5))
sns.boxplot(x='label', y='comment_ratio', data=df)
plt.title("Comment Ratio by Label")
plt.show()

# --- Correlation Heatmap
plt.figure(figsize=(10,7))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Numeric Features")
plt.show()

# =====================
# 3. Train-Test Split
# =====================

# Choose features for ML (numeric + text length + engagement ratios)
X = df[['video_view_count_log', 'video_like_count_log', 'video_comment_count_log',
        'title_length', 'desc_length', 'transcript_length', 'like_ratio', 'comment_ratio']]

# Target variable
y = df['label']

# Train-Test Split (80% train, 20% test, stratified to balance labels)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n Train-Test Split Complete")
print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)
print("Train Label Distribution:\n", y_train.value_counts(normalize=True))
print("Test Label Distribution:\n", y_test.value_counts(normalize=True))
