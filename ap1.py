import pandas as pd
import re
import string
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier # Updated for modern use
from sklearn.metrics import accuracy_score, classification_report

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)

# 1. LOAD THE DATASET
try:
    data = pd.read_csv("cleaned_fake_news.csv")
except FileNotFoundError:
    data = pd.read_csv("filtered_fake_news_dataset.csv")
    if 'title' in data.columns and 'text' not in data.columns:
        data = data.rename(columns={'title': 'text'})

# 2. ADVANCED CLEANING
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text) 
    text = re.sub(r'<.*?>', '', text)                
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text) 
    text = re.sub(r'\d+', '', text)                 
    text = re.sub(r'\s+', ' ', text).strip()        
    return text

data['text'] = data['text'].fillna('missing').apply(clean_text)
data = data[data['text'] != ""] 

# 3. SPLIT THE DATA
x_train, x_test, y_train, y_test = train_test_split(
    data['text'], 
    data['label'], 
    test_size=0.2, 
    random_state=42,
    stratify=data['label']
)

# 4. CONVERT TEXT TO NUMBERS
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, max_features=10000, ngram_range=(1,2))
tfidf_train = tfidf_vectorizer.fit_transform(x_train) 
tfidf_test = tfidf_vectorizer.transform(x_test)

# 5. HYPERPARAMETER TUNING (Updated to SGDClassifier)
# This mimics the Passive Aggressive behavior but uses the modern standard
param_grid = {'alpha': [0.0001, 0.001, 0.01]}
grid_search = GridSearchCV(
    SGDClassifier(loss='hinge', penalty=None, learning_rate='pa1', eta0=1.0, max_iter=2000), 
    param_grid, 
    cv=5
)
grid_search.fit(tfidf_train, y_train)

best_model = grid_search.best_estimator_

# 6. EVALUATE
y_pred = best_model.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)

print(f"--- Optimized Model Summary ---")
print(f"Accuracy Score: {round(score*100, 2)}%")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# 7. PREDICTION FUNCTION
def predict_news(news_item):
    cleaned = clean_text(news_item)
    vectorized = tfidf_vectorizer.transform([cleaned])
    return best_model.predict(vectorized)[0]

# Manual Test
print("-" * 30)
sample = "Scientific breakthrough in quantum computing announced"
print(f"Test Sentence: {sample}")
print(f"Prediction: {predict_news(sample)}")

# Manual Test
print("-" * 30)
sample = "Scientific breakthrough in quantum computing announced"
print(f"Test Sentence: {sample}")
print(f"Prediction: {predict_news(sample)}")

# ================= SAVE MODEL =================
import pickle

pickle.dump(best_model, open("model.pkl", "wb"))
pickle.dump(tfidf_vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model and vectorizer saved successfully")