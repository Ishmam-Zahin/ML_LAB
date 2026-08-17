import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler

os.makedirs('images', exist_ok=True)

# ── 1. Dataset — make_circles ─────────────────────────────────────────
X, y = make_circles(n_samples=500, noise=0.05, factor=0.4, random_state=42)

scaler = StandardScaler()
X = scaler.fit_transform(X)

plt.figure(figsize=(5, 4))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
plt.title('Dataset: Concentric Circles (true labels)')
plt.tight_layout()
plt.savefig('images/circles_dataset.png')
plt.show()


# ── 2. K-Means ────────────────────────────────────────────────────────
kmeans   = KMeans(n_clusters=2, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X)

ari_km = adjusted_rand_score(y, y_kmeans)
nmi_km = normalized_mutual_info_score(y, y_kmeans)


# ── 3. Hierarchical ───────────────────────────────────────────────────
hier   = AgglomerativeClustering(n_clusters=2, linkage='ward')
y_hier = hier.fit_predict(X)

ari_hc = adjusted_rand_score(y, y_hier)
nmi_hc = normalized_mutual_info_score(y, y_hier)


# ── 4. DBSCAN ─────────────────────────────────────────────────────────
dbscan   = DBSCAN(eps=0.3, min_samples=5)
y_dbscan = dbscan.fit_predict(X)

ari_db = adjusted_rand_score(y, y_dbscan)
nmi_db = normalized_mutual_info_score(y, y_dbscan)


# ── 5. Four-panel cluster comparison ──────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(18, 4))

plots = [
    (y,        'Ground Truth'),
    (y_kmeans, f'K-Means\nARI={ari_km:.3f}  NMI={nmi_km:.3f}'),
    (y_hier,   f'Hierarchical\nARI={ari_hc:.3f}  NMI={nmi_hc:.3f}'),
    (y_dbscan, f'DBSCAN\nARI={ari_db:.3f}  NMI={nmi_db:.3f}'),
]

for ax, (labels, title) in zip(axes, plots):
    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='coolwarm', s=20, alpha=0.8)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle('Clustering Comparison — Concentric Circles', y=1.02)
plt.tight_layout()
plt.savefig('images/all_comparison.png', bbox_inches='tight')
plt.show()


# ── 6. Dendrogram ─────────────────────────────────────────────────────
Z = linkage(X[:80], method='ward')

plt.figure(figsize=(14, 5))
dendrogram(Z, leaf_rotation=90, leaf_font_size=7, color_threshold=Z[-2, 2])
plt.axhline(y=Z[-2, 2], color='red', linestyle='--', label='Cut for k=2')
plt.title('Dendrogram (Ward linkage, 80-sample subset)')
plt.xlabel('Sample index')
plt.ylabel('Merge distance')
plt.legend()
plt.tight_layout()
plt.savefig('images/dendrogram.png')
plt.show()


# ── 7. Metrics bar chart ──────────────────────────────────────────────
methods = ['K-Means', 'Hierarchical', 'DBSCAN']
aris    = [ari_km,    ari_hc,         ari_db]
nmis    = [nmi_km,    nmi_hc,         nmi_db]

x = np.arange(len(methods))
w = 0.35

fig, ax = plt.subplots(figsize=(7, 4))
b1 = ax.bar(x - w/2, aris, w, label='ARI')
b2 = ax.bar(x + w/2, nmis, w, label='NMI')
ax.bar_label(b1, fmt='%.3f', padding=3)
ax.bar_label(b2, fmt='%.3f', padding=3)
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.set_ylabel('Score')
ax.set_ylim(0, 1.15)
ax.set_title('ARI & NMI — All Methods')
ax.legend()
plt.tight_layout()
plt.savefig('images/metrics_bar.png')
plt.show()


# ── 8. Print table ────────────────────────────────────────────────────
print(f"\n{'Method':<20} {'ARI':>6}  {'NMI':>6}")
print("-" * 36)
for m, a, n in zip(methods, aris, nmis):
    print(f"{m:<20} {a:>6.3f}  {n:>6.3f}")