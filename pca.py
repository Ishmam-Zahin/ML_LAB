import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

np.random.seed(42)

# ---- 1. Generate dataset with 5 features, 4 clusters ----
X, y = make_blobs(n_samples=1000, centers=4, n_features=5, random_state=42, cluster_std=2.0)

# ---- 2. Plot the raw dataset (using first 2 features for a quick look) ----
plt.figure(figsize=(5, 5))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=15)
plt.title('Dataset (features 0 vs 1)')
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.tight_layout()
plt.savefig('images/dataset_5d.png', dpi=120)
plt.close()

# ---- 3. KMeans directly on the 5D data ----
model_5d = KMeans(n_clusters=4, random_state=42, n_init=10)
labels_5d = model_5d.fit_predict(X)
centers_5d = model_5d.cluster_centers_

ari_5d = adjusted_rand_score(y, labels_5d)
nmi_5d = normalized_mutual_info_score(y, labels_5d)

# ---- 4. PCA to reduce 5D -> 2D ----
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

# ---- 5. KMeans on the 2D PCA-reduced data ----
model_pca = KMeans(n_clusters=4, random_state=42, n_init=10)
labels_pca = model_pca.fit_predict(X_pca)
centers_pca = model_pca.cluster_centers_

ari_pca = adjusted_rand_score(y, labels_pca)
nmi_pca = normalized_mutual_info_score(y, labels_pca)

print(f"Explained variance ratio (PC1, PC2): {pca.explained_variance_ratio_}")
print(f"Total variance kept: {pca.explained_variance_ratio_.sum():.3f}")
print()
print("=== KMeans on original 5D data ===")
print(f"ARI: {ari_5d:.4f}")
print(f"NMI: {nmi_5d:.4f}")
print()
print("=== KMeans on PCA-reduced 2D data ===")
print(f"ARI: {ari_pca:.4f}")
print(f"NMI: {nmi_pca:.4f}")

# ---- 6. Plot clusters + centers, 5D case projected via PCA for visualization ----
centers_5d_proj = pca.transform(centers_5d)  # project 5D centers into PCA space just to visualize

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels_5d, cmap='viridis', s=15)
axes[0].scatter(centers_5d_proj[:, 0], centers_5d_proj[:, 1], marker='X', c='red', s=200, edgecolor='black')
axes[0].set_title(f'KMeans fit on 5D data\n(shown via PCA projection)\nARI={ari_5d:.3f}, NMI={nmi_5d:.3f}')
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')

axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=labels_pca, cmap='viridis', s=15)
axes[1].scatter(centers_pca[:, 0], centers_pca[:, 1], marker='X', c='red', s=200, edgecolor='black')
axes[1].set_title(f'KMeans fit on PCA(2D) data\nARI={ari_pca:.3f}, NMI={nmi_pca:.3f}')
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')

plt.tight_layout()
plt.savefig('images/clusters_comparison.png', dpi=120)
plt.close()

# ---- 7. Plot just the PCA-reduced dataset colored by true labels ----
plt.figure(figsize=(5, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=15)
plt.title('PCA-reduced dataset (true labels)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.tight_layout()
plt.savefig('images/pca_dataset.png', dpi=120)
plt.close()

print("\nSaved: images/dataset_5d.png, images/clusters_comparison.png, images/pca_dataset.png")