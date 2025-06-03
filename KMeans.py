import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import rand_score, adjusted_rand_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import umap



npy_path = 'data/feat_DINO_VocPascal.npy'
txt_path = 'VocPascal/val_voc.txt'
reduce_method = 'umap' 
reduce_dim = 2


X = np.load(npy_path) 


with open(txt_path, 'r') as f:
    lines = [line.strip().split('\t') for line in f]

file_names = [line[0] for line in lines]
true_labels = [line[1] for line in lines]


unique_labels = sorted(set(true_labels))
label_to_int = {label: i for i, label in enumerate(unique_labels)}
true_labels_int = np.array([label_to_int[label] for label in true_labels])


if reduce_method == 'pca':
    reducer = PCA(n_components=reduce_dim)
    X_reduced = reducer.fit_transform(X)
elif reduce_method == 'umap':
    
    reducer = umap.UMAP(n_components=reduce_dim)
    X_reduced = reducer.fit_transform(X)
elif reduce_method=="not_reduced":
    print("entre")
    X_reduced = X



k = 50
kmeans = KMeans(n_clusters=k)
kmeans.fit(X_reduced)
predicted_labels = kmeans.labels_
centroids = kmeans.cluster_centers_
print(f"Número de centroides: {kmeans.n_clusters}")

ri = rand_score(true_labels_int, predicted_labels)
ari = adjusted_rand_score(true_labels_int, predicted_labels)
print(f"Rand Index: {ri:.4f}")
print(f"Adjusted Rand Index: {ari:.4f}")

np.random.seed(0)
sample_indices = np.random.choice(len(X_reduced), size=100, replace=False)
X_sampled = X_reduced[sample_indices]
pred_sampled = np.array(predicted_labels)[sample_indices]
true_sampled = np.array(true_labels_int)[sample_indices]


plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(X_sampled[:, 0], X_sampled[:, 1], c=pred_sampled, cmap='tab10', s=40)
plt.title("Agrupamiento K-Means (100 muestras)")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")

plt.subplot(1, 2, 2)
plt.scatter(X_sampled[:, 0], X_sampled[:, 1], c=true_sampled, cmap='tab10', s=40)
plt.title("Etiquetas Reales (100 muestras)")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.tight_layout()
plt.show()