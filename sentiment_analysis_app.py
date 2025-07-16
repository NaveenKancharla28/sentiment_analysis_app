# sentiment_analysis_app.py

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Load and rename columns for Sentiment140
df = pd.read_csv('sentiment_data.csv', encoding='ISO-8859-1', header=None)

df.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']
df['sentiment'] = df['sentiment'].map({0: 0, 4: 1})  # Binary sentiment

# Optional: use smaller sample for faster testing
df = df.sample(20000, random_state=42)

# Clean text
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
    return text

df['cleaned_text'] = df['text'].apply(clean_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_text']).toarray()
y = df['sentiment']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Predict & evaluate
preds = clf.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, preds))
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, preds) * 100))

# Plot sentiment distribution
plt.figure(figsize=(6, 4))
df['sentiment'].value_counts().plot(kind='bar', color=['red', 'green'])
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.xticks(ticks=[0, 1], labels=['Negative', 'Positive'], rotation=0)
plt.tight_layout()
plt.show()