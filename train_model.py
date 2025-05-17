import pandas as pd 
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset
data = pd.read_csv('fitness_data.csv')

# Features and target (7 features)
X = data[['sleep', 'stress', 'fatigue', 'motivation', 'nutrition', 'soreness', 'mood']]
y = data['training_level']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize and train the Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(report)

# Test cases
test_cases = np.array([
    [5, 1, 1, 5, 5, 0, 5],  # Excellent state -> Go Hard
    [4, 2, 2, 4, 4, 1, 4],  # Good state -> Moderate Training
    [3, 3, 3, 3, 3, 2, 3],  # Average state -> Light Training
    [2, 4, 4, 2, 2, 3, 2],  # Poor state -> Rest Day
    [1, 5, 5, 1, 1, 4, 1]   # Very poor -> Strict Rest (No Training)
])

predictions = model.predict(test_cases)
print("\nTest Case Predictions:")
for i, case in enumerate(test_cases):
    print(f"Case {i+1} {case}: Level {predictions[i]}")

# Save the model (IMPORTANT: same filename as used in app.py)
joblib.dump(model, 'models/naive_bayes.pkl')
print("✅ Model saved as models/naive_bayes.pkl")


