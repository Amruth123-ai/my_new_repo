# this is the classification predection model were it classifices the data using SVM 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# 1. Load cleaned CSV
df = pd.read_csv('/content/sensitive_dataset_1L.csv')  # Update if needed
print("Sample data:\n", df.head())

# 2. Prepare features and labels
X = df['text']
y = df['category']

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# 4. Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5. Train SVM Classifier (with probability support)
clf = SVC(kernel='linear', probability=True, random_state=42)
clf.fit(X_train_vec, y_train)

# 6. Evaluate
y_pred = clf.predict(X_test_vec)
print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
#print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# 7. Prediction function with thresholding
def predict_category(text, threshold=0.6):
    vec = vectorizer.transform([text])
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(vec)[0]
        max_prob = np.max(probs)
        pred_label = clf.predict(vec)[0]
        print(f"\nText: '{text}'\nProbabilities: {dict(zip(clf.classes_, probs))}\nMax prob: {max_prob:.4f}")
        if max_prob < threshold:
            return "category not related"
        return pred_label
    else:
        return clf.predict(vec)[0]

# 8. Test on sample inputs
sample_texts = [
    "i want an apple fruit",
    "Apple released a new MacBook.",
    "i want to buy an Apple iphone",
    "hanumansai need an Apple fruit",
    "is Apple iphone worthy?",
    "my favorite fruit is apple",
    "The sky is blue today.",
    "Microsoft announces new Surface laptop",
    "Fresh apples are in season.",
    "The CEO of Apple gave a keynote speech.",
    "I'm going on a vacation next week.",
    "hanumansai favarite drink is apple juice"
]

for text in sample_texts:
    result = predict_category(text)
    print(f"🔍 '{text}' → Predicted: {result}")
