import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load the reshaped dataset
df = pd.read_csv('dataset.csv')

# Split the data into features (ciphertexts) and labels (algorithms)
X = df['Ciphertext']  # Features (Ciphertext)
y = df['Algorithm']   # Labels (Algorithm)

# Split into training and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert ciphertexts into numerical features using TF-IDF
tfidf_vectorizer = TfidfVectorizer(ngram_range=(2, 4), analyzer='char')  # character-level n-grams
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Build the classification model (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model and the TF-IDF vectorizer to disk
joblib.dump(model, 'cipher_classification_model.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
print('Model and vectorizer saved to disk.')