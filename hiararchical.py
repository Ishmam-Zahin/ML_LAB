import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.cluster.hierarchy import dendrogram, linkage

# ── 1. Dataset — make_moons (K-Means fails here) ──────────────────────
# K-Means assumes spherical clusters; moons are non-convex → it breaks
X, y = make_moons(n_samples=500, noise=0.1, random_state=42)

plt.figure(figsize=(5, 4))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
plt.title('Dataset: Two Moons (true labels)')
plt.tight_layout()
plt.savefig('images/moons_dataset.png')


# ── 2. K-Means on moons ───────────────────────────────────────────────
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X)

ari_km  = adjusted_rand_score(y, y_kmeans)
nmi_km  = normalized_mutual_info_score(y, y_kmeans)


# ── 3. Hierarchical (Agglomerative) on moons ─────────────────────────
# linkage='ward'  → minimizes within-cluster variance at each merge
# linkage='single' / 'complete' / 'average' are alternatives
hier = AgglomerativeClustering(n_clusters=2, linkage='ward')
y_hier = hier.fit_predict(X)

ari_hc  = adjusted_rand_score(y, y_hier)
nmi_hc  = normalized_mutual_info_score(y, y_hier)


# ── 4. Side-by-side cluster plots ────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
axes[0].set_title('Ground Truth')

axes[1].scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='coolwarm')
axes[1].set_title(f'K-Means\nARI={ari_km:.3f}  NMI={nmi_km:.3f}')

axes[2].scatter(X[:, 0], X[:, 1], c=y_hier, cmap='coolwarm')
axes[2].set_title(f'Hierarchical (Ward)\nARI={ari_hc:.3f}  NMI={nmi_hc:.3f}')

plt.tight_layout()
plt.savefig('images/comparison.png')


# ── 5. Dendrogram ─────────────────────────────────────────────────────
# scipy linkage on a subsample — full 500 pts makes it unreadable
X_sub = X[:80]                         # 80 samples is enough to see the tree
Z     = linkage(X_sub, method='ward')  # Z shape: (n-1, 4) — merge history

plt.figure(figsize=(14, 5))
dendrogram(
    Z,
    leaf_rotation=90,
    leaf_font_size=7,
    color_threshold=Z[-2, 2],   # color clusters at the 2-cluster cut
)
# plt.axhline(y=Z[-2, 2], color='red', linestyle='--', label='Cut for k=2')
plt.title('Dendrogram (Ward linkage, 80-sample subset)')
plt.xlabel('Sample index')
plt.ylabel('Merge distance')
plt.legend()
plt.tight_layout()
plt.savefig('images/dendrogram.png')


# ── 6. Metrics summary ────────────────────────────────────────────────
print(f"{'Method':<20} {'ARI':>6}  {'NMI':>6}")
print("-" * 36)
print(f"{'K-Means':<20} {ari_km:>6.3f}  {nmi_km:>6.3f}")
print(f"{'Hierarchical (Ward)':<20} {ari_hc:>6.3f}  {nmi_hc:>6.3f}")