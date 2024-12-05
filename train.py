import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# obstacle_position veriseti
obstacle_positions = [[0, 0], [1, 1], [2, 2], [3, 3]]

# Engel pozisyonlarını ölçeklendirin
scaler = StandardScaler()
obstacle_positions_scaled = scaler.fit_transform(obstacle_positions)

# Scaler modelini kaydet
joblib.dump(scaler, 'scaler.joblib')

# KMeans modelini eğitin
n_clusters = 1
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(obstacle_positions_scaled)

# KMeans modelini kaydet
joblib.dump(kmeans, 'kmeans_model.joblib')

print("Scaler ve KMeans modeli başarıyla kaydedildi.")
