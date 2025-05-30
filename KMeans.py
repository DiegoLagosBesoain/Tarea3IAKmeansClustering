import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# === Configuración ===
npy_path = 'data/feat_DINO_VocPascal.npy'  # tu archivo .npy
txt_path = 'VocPascal/train_voc.txt'               # tu archivo .txt con nombres y etiquetas

# === 1. Cargar los datos ===
X = np.load(npy_path)  # (n_samples, n_features)

# === 2. Leer los nombres y etiquetas ===
with open(txt_path, 'r') as f:
    lines = [line.strip().split('\t') for line in f]

file_names = [line[0] for line in lines]
true_labels = [line[1] for line in lines]

# === 3. Convertir etiquetas reales a números si son strings ===
unique_labels = sorted(set(true_labels))
label_to_int = {label: i for i, label in enumerate(unique_labels)}
true_labels_int = [label_to_int[label] for label in true_labels]

# === 4. K-Means ===
k = len(set(true_labels_int))  # o elegirlo manualmente
kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(X)

predicted_labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# === 5. Graficar clustering (K-Means) ===
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=predicted_labels, cmap='tab10', s=40)
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=100, label='Centroides')
plt.title("Agrupamiento K-Means")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.legend()

# === 6. Graficar etiquetas reales ===
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=true_labels_int, cmap='tab10', s=40)
plt.title("Etiquetas Reales")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")

plt.tight_layout()
plt.show()