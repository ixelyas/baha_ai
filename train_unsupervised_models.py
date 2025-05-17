# ----- UNSUPERVISED -----
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import joblib

# Load data
data = pd.read_csv('fitness_data.csv')
X = data[['sleep', 'stress', 'fatigue', 'motivation', 'nutrition', 'soreness', 'mood']]

# KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
joblib.dump(kmeans, 'models/kmeans.pkl')

# PCA
pca = PCA(n_components=2)
pca.fit(X)
joblib.dump(pca, 'models/pca.pkl')

# Apriori — pseudo rules
rules = [
    "Fatigue >= 4 → Rest Day or Light Training",
    "Motivation <= 2 → Rest Day",
    "Soreness >= 4 → Light Training or Rest",
    "Sleep <= 2 and Stress >= 4 → Strict Rest (No Training)",
    "Sleep >= 4 and Motivation >= 4 → Go Hard"
]

joblib.dump(rules, 'models/apriori.pkl')

print("✅ Unsupervised модели обучены и сохранены в models/")
