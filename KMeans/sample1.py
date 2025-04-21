import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#müşteri gelir ve harcama listesi
X = np.array([
    [15,39],[16,50],[25,5],[85,59],[89,60],[75,39],[10,8],[150,29],[130,19],[24,79],[88,62],[85,49],[85,45],
])

kmeans = KMeans(n_clusters=4, random_state=42)

kmeans.fit(X)
labels = kmeans.labels_

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, marker='X', c='black')
plt.xlabel("Gelir")
plt.ylabel("Harcama")
plt.title("K-means ile Müşteri Segmentasyonu")
plt.show()