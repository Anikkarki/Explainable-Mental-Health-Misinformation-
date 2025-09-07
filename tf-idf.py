import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset
df = pd.read_csv("videos_MHMisinfo_Prepared.csv")

# 2. Combine text fields into one input
df["text"] = (
    df["video_title"].fillna("") + " " +
    df["video_description"].fillna("") + " " +
    df["audio_transcript"].fillna("")
)

# 3. Define features and labels
X = df["text"]
y = df["label"]   # 'reliable' or 'misinformation'

# 4. Train/test split (stratified to preserve imbalance ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Vectorize with TF-IDF
vectorizer = TfidfVectorizer(
    stop_words="english", 
    max_features=20000, 
    ngram_range=(1,2)   # unigrams + bigrams
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 6. Train Logistic Regression
clf = LogisticRegression(max_iter=2000, class_weight="balanced")
clf.fit(X_train_tfidf, y_train)

# 7. Evaluate
y_pred = clf.predict(X_test_tfidf)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# 8. Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=clf.classes_, yticklabels=clf.classes_, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - TF-IDF + Logistic Regression")
plt.show()

# 9. Top indicative words
feature_names = vectorizer.get_feature_names_out()
coefs = clf.coef_[0]
top_misinfo_idx = coefs.argsort()[-15:][::-1]
top_reliable_idx = coefs.argsort()[:15]

print("\nTop words indicating MISINFORMATION:")
for i in top_misinfo_idx:
    print(f"{feature_names[i]} ({coefs[i]:.3f})")

print("\nTop words indicating RELIABLE:")
for i in top_reliable_idx:
    print(f"{feature_names[i]} ({coefs[i]:.3f})")
