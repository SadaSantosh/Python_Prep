import pandas as pd
import numpy as np
import re
import string
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Synthetic Dataset Creation (SMS/Email Spam & Phishing Messages)
data = {
    'text': [
        "URGENT! Your bank account has been suspended. Click http://bit.ly/secure-login to verify now!",
        "Hey, are we still meeting for lunch today at 1 PM?",
        "Congratulations! You won a $1000 Walmart gift card! Claim here: http://claim-prize.online",
        "Can you send over the updated project slides when you get a chance?",
        "Security Alert: Unusual login attempt detected. Verify your password immediately at http://fake-auth.com",
        "Don't forget to buy groceries on your way back home.",
        "WINNER! Free iPhone 15 Pro Max giveaway. Click link to claim instantly!",
        "The meeting has been rescheduled to tomorrow morning at 10 AM.",
        "Your package delivery failed. Please update your address at http://track-parcel-fix.com",
        "Thanks for helping out yesterday, really appreciated it!"
    ],
    'label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham']
}

# Duplicate dataset to create a balanced synthetic training set
df = pd.DataFrame(data)
df = pd.concat([df] * 15, ignore_index=True)

# 2. Text Preprocessing Function
def clean_text(text):
    text = text.lower() # Lowercasing
    text = re.sub(r'http\S+|www\S+|https\S+', 'http_link', text) # Standardize URLs
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text) # Remove Punctuation
    text = re.sub(r'\d+', '', text) # Remove Numbers
    return text

df['cleaned_text'] = df['text'].apply(clean_text)

# 3. TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
X = tfidf.fit_transform(df['cleaned_text'])
y = df['label']

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train Naive Bayes Classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# 6. Model Evaluation
y_pred = model.predict(X_test)
print(f"✅ Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7. Save Artifacts
joblib.dump(model, 'spam_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
print("\n💾 Model and Vectorizer saved successfully!")