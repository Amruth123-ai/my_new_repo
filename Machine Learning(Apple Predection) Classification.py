# creating a dataseet for the classification predection

import random
import csv

# Define sensitive templates for each category
apple_tech_templates = [
    "Apple launched a new {product}.",
    "I want to buy an Apple {product}.",
    "Apple's {product} has a {feature}.",
    "Apple introduced the {chip} chip.",
    "Apple opened a store in {city}.",
    "Apple's stock {stock_action} after {event}.",
    "Apple's {service} got an update.",
    "Apple acquired a {industry} startup.",
    "I love my new Apple {product}.",
    "Apple's {product} supports {feature}.",
    "Apple announced {software} for devices.",
    "Apple's {product} is great for {activity}.",
    "Apple is developing {technology} tech.",
    "Apple's {product} comes in {color}.",
    "I need an Apple {product} for {purpose}."
]

apple_fruit_templates = [
    "I want an apple fruit.",
    "I ate a {variety} apple fruit.",
    "She made an apple {dish} with fruit.",
    "Apple {beverage} from fresh fruit is tasty.",
    "I picked an apple fruit from the {place}.",
    "An apple fruit a day is healthy.",
    "I baked apple fruit with {ingredient}.",
    "He brought {variety} apple fruit for {meal}.",
    "Apple fruit trees grow in the {place}.",
    "The market sells {type} apple fruit.",
    "She sliced apple fruit for a {dish}.",
    "I bought {type} apple fruit at the store.",
    "Apple fruit is great for {activity}.",
    "I planted an apple fruit tree in {place}.",
    "Fresh apple fruit smells {adjective}.",
    "my favarite drink is apple {juice}"

]

unrelated_templates = [
    "The {weather} is {condition} today.",
    "I'm planning a trip to {city}.",
    "{company} released a new {product}.",
    "The {animal} {action} in the {place}.",
    "I love {activity} during {season}.",
    "{sport} season starts {time}.",
    "{company} is working on {technology}.",
    "The {media} was {adjective} and {adjective}.",
    "I bought a {item} at the store.",
    "The {event} was a huge success.",
    "I'm learning to {activity}.",
    "The {place} is beautiful in {season}.",
    "I watched a {media} about {topic}.",
    "The {food} was delicious at the {place}.",
    "I'm excited for the {event}."
]

# Keywords for filling templates (carefully chosen to avoid overlap)
apple_tech_keywords = {
    "product": ["iPhone", "iPad", "MacBook", "Apple Watch", "AirPods", "Vision Pro", "Mac Mini", "iMac"],
    "feature": ["faster processor", "better camera", "OLED display", "longer battery", "AI support", "5G connectivity"],
    "chip": ["M3", "M4", "A18", "A19"],
    "city": ["San Francisco", "Tokyo", "London", "Sydney", "Paris", "Dubai", "New York", "Chicago"],
    "stock_action": ["rose", "fell", "surged", "dipped"],
    "event": ["keynote", "product launch", "earnings report"],
    "service": ["iCloud", "Apple Music", "Apple TV", "Fitness+"],
    "industry": ["AI", "gaming", "health tech", "fintech"],
    "software": ["iOS", "macOS", "watchOS", "iPadOS"],
    "activity": ["work", "gaming", "studying", "fitness"],
    "technology": ["augmented reality", "machine learning", "virtual reality"],
    "color": ["space gray", "silver", "midnight blue", "rose gold"],
    "purpose": ["work", "school", "gaming", "travel"]
}

apple_fruit_keywords = {
    "variety": ["Fuji", "Granny Smith", "Honeycrisp", "Gala", "McIntosh", "Pink Lady", "Braeburn"],
    "dish": ["pie", "tart", "sauce", "crumble", "salad", "jam"],
    "beverage": ["juice", "cider"],
    "place": ["orchard", "garden", "farm", "backyard"],
    "ingredient": ["cinnamon", "honey", "sugar", "nutmeg"],
    "meal": ["breakfast", "lunch", "snack", "dinner"],
    "type": ["organic", "fresh", "crisp", "juicy"],
    "activity": ["baking", "eating", "cooking", "snacking"],
    "adjective": ["sweet", "fresh", "crisp", "delicious"]
}

unrelated_keywords = {
    "weather": ["sky", "weather", "sunset"],
    "condition": ["clear", "cloudy", "sunny", "rainy"],
    "city": ["Paris", "Tokyo", "London", "New York", "Sydney"],
    "company": ["Microsoft", "Google", "Tesla", "Amazon", "Netflix"],
    "product": ["Windows", "car", "smartphone", "TV series"],
    "animal": ["cat", "dog", "bird"],
    "action": ["ran", "slept", "played"],
    "place": ["park", "house", "yard"],
    "activity": ["swimming", "hiking", "reading", "painting"],
    "season": ["summer", "winter", "fall", "spring"],
    "sport": ["soccer", "tennis", "basketball"],
    "time": ["next week", "soon", "tomorrow"],
    "technology": ["blockchain", "quantum computing"],
    "media": ["movie", "novel"],
    "adjective": ["exciting", "boring", "funny", "sad"],
    "item": ["book", "shirt", "phone"],
    "event": ["concert", "festival", "game"],
    "topic": ["space", "history", "nature"],
    "food": ["pizza", "sushi", "tacos"]
}

# Function to generate a sentence
def generate_sentence(template, keywords):
    sentence = template
    for key in keywords:
        sentence = sentence.replace(f"{{{key}}}", random.choice(keywords[key]))
    return sentence

# Generate dataset
dataset = []
target_per_category = 33333  # Approx 5000 / 3

# Generate Apple_tech samples
for _ in range(target_per_category):
    template = random.choice(apple_tech_templates)
    sentence = generate_sentence(template, apple_tech_keywords)
    dataset.append({"text": sentence, "category": "Apple_tech"})

# Generate Apple_fruit samples
for _ in range(target_per_category):
    template = random.choice(apple_fruit_templates)
    sentence = generate_sentence(template, apple_fruit_keywords)
    dataset.append({"text": sentence, "category": "Apple_fruit"})

# Generate Unrelated samples
for _ in range(target_per_category - 2):  # Adjust for exactly 5000
    template = random.choice(unrelated_templates)
    sentence = generate_sentence(template, unrelated_keywords)
    dataset.append({"text": sentence, "category": "Unrelated"})

# Add specific problematic examples with correct labels
dataset.append({"text": "I want an Apple iPhone.", "category": "Apple_tech"})
dataset.append({"text": "I want an apple fruit.", "category": "Apple_fruit"})

# Shuffle the dataset
random.shuffle(dataset)

# Save to CSV with correct file mode
with open("sensitive_dataset_1L.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=["text", "category"])
    writer.writeheader()
    for row in dataset:
        writer.writerow(row)

print("Sensitive dataset with 5000 samples generated and saved as 'sensitive_dataset_1L.csv'.")

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------

# this is the (Machine Learning) classification predection model were it classifices the data(company/fruit) using SVM 

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

# output will be :

# Text: 'i want an apple fruit'
# Probabilities: {'Apple_fruit': np.float64(0.9999522626044371), 'Apple_tech': np.float64(4.7002668603707204e-05), 'Unrelated': np.float64(7.34726959008475e-07)}
# Max prob: 1.0000
# 🔍 'i want an apple fruit' → Predicted: Apple_fruit

# Text: 'Apple released a new MacBook.'
# Probabilities: {'Apple_fruit': np.float64(0.0008191685227575213), 'Apple_tech': np.float64(0.9666677230846816), 'Unrelated': np.float64(0.03251310839256085)}
# Max prob: 0.9667
# 🔍 'Apple released a new MacBook.' → Predicted: Apple_tech

