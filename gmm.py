import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler

os.makedirs('images', exist_ok=True)

# ── 1. Dataset — Anisotropic (elongated) Gaussians ────────────────────
# GMM models full covariance → handles elliptical shapes perfectly
# K-Means assumes spherical clusters → fails on elongated blobs
# DBSCAN needs uniform density → fails when clusters overlap
X, y = make_blobs(n_samples=500, centers=4, random_state=42)

# Apply linear transformation to stretch and rotate clusters
transform = np.array([[0.6, -0.7],
                       [0.3,  0.9]])
X = X @ transform

scaler = StandardScaler()
X = scaler.fit_transform(X)

plt.figure(figsize=(5, 4))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='tab10', s=20)
plt.title('Dataset: Anisotropic Gaussians (true labels)')
plt.tight_layout()
plt.savefig('images/gmm_dataset.png')
plt.show()


# ── 2. K-Means ────────────────────────────────────────────────────────
kmeans   = KMeans(n_clusters=4, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X)

ari_km = adjusted_rand_score(y, y_kmeans)
nmi_km = normalized_mutual_info_score(y, y_kmeans)


# ── 3. Hierarchical ───────────────────────────────────────────────────
hier   = AgglomerativeClustering(n_clusters=4, linkage='ward')
y_hier = hier.fit_predict(X)

ari_hc = adjusted_rand_score(y, y_hier)
nmi_hc = normalized_mutual_info_score(y, y_hier)


# ── 4. DBSCAN ─────────────────────────────────────────────────────────
dbscan   = DBSCAN(eps=0.4, min_samples=5)
y_dbscan = dbscan.fit_predict(X)

ari_db = adjusted_rand_score(y, y_dbscan)
nmi_db = normalized_mutual_info_score(y, y_dbscan)


# ── 5. GMM ────────────────────────────────────────────────────────────
# covariance_type='full' → each cluster gets its own ellipse shape
# covariance_type='spherical' → same as K-Means assumption (worse here)
gmm   = GaussianMixture(n_components=4, covariance_type='full', random_state=42)
gmm.fit(X)
y_gmm = gmm.predict(X)

ari_gmm = adjusted_rand_score(y, y_gmm)
nmi_gmm = normalized_mutual_info_score(y, y_gmm)


# ── 6. Five-panel cluster comparison ──────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(22, 4))

plots = [
    (y,        'Ground Truth'),
    (y_kmeans, f'K-Means\nARI={ari_km:.3f}  NMI={nmi_km:.3f}'),
    (y_hier,   f'Hierarchical\nARI={ari_hc:.3f}  NMI={nmi_hc:.3f}'),
    (y_dbscan, f'DBSCAN\nARI={ari_db:.3f}  NMI={nmi_db:.3f}'),
    (y_gmm,    f'GMM\nARI={ari_gmm:.3f}  NMI={nmi_gmm:.3f}'),
]

for ax, (labels, title) in zip(axes, plots):
    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', s=20, alpha=0.8)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle('Clustering Comparison — Anisotropic Gaussians', y=1.02)
plt.tight_layout()
plt.savefig('images/all_comparison.png', bbox_inches='tight')
plt.show()


# ── 7. Dendrogram ─────────────────────────────────────────────────────
Z = linkage(X[:80], method='ward')

plt.figure(figsize=(14, 5))
dendrogram(Z, leaf_rotation=90, leaf_font_size=7, color_threshold=Z[-4, 2])
plt.axhline(y=Z[-4, 2], color='red', linestyle='--', label='Cut for k=4')
plt.title('Dendrogram (Ward linkage, 80-sample subset)')
plt.xlabel('Sample index')
plt.ylabel('Merge distance')
plt.legend()
plt.tight_layout()
plt.savefig('images/dendrogram.png')
plt.show()


# ── 8. GMM ellipses plot ──────────────────────────────────────────────
# Visualize the covariance ellipses GMM learned per cluster
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(X[:, 0], X[:, 1], c=y_gmm, cmap='tab10', s=20, alpha=0.5)

for i in range(gmm.n_components):
    mean = gmm.means_[i]
    cov  = gmm.covariances_[i]

    # Eigendecomposition → ellipse axes
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    angle  = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * 2 * np.sqrt(vals)   # 2-sigma ellipse

    from matplotlib.patches import Ellipse
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                      edgecolor='black', fc='none', lw=2)
    ax.add_patch(ellipse)
    ax.plot(*mean, 'kx', markersize=10, markeredgewidth=2)

ax.set_title('GMM — Learned Covariance Ellipses (2σ)')
ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()
plt.savefig('images/gmm_ellipses.png')
plt.show()


# ── 9. BIC curve to pick optimal n_components ─────────────────────────
# BIC (Bayesian Information Criterion) — lower is better
# Equivalent of the elbow plot for GMM
bics = []
k_range = range(1, 11)

for k in k_range:
    g = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
    g.fit(X)
    bics.append(g.bic(X))

plt.figure(figsize=(6, 4))
plt.plot(k_range, bics, marker='o', linewidth=2)
plt.axvline(x=4, color='red', linestyle='--', label='True k=4')
plt.xlabel('Number of components (k)')
plt.ylabel('BIC (lower = better)')
plt.title('BIC Curve — GMM (equivalent of elbow for K-Means)')
plt.xticks(k_range)
plt.legend()
plt.tight_layout()
plt.savefig('images/bic_curve.png')
plt.show()


# ── 10. Metrics bar chart ─────────────────────────────────────────────
methods = ['K-Means', 'Hierarchical', 'DBSCAN',  'GMM']
aris    = [ari_km,    ari_hc,         ari_db,     ari_gmm]
nmis    = [nmi_km,    nmi_hc,         nmi_db,     nmi_gmm]

x = np.arange(len(methods))
w = 0.35

fig, ax = plt.subplots(figsize=(8, 4))
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


# ── 11. Print table ───────────────────────────────────────────────────
print(f"\n{'Method':<20} {'ARI':>6}  {'NMI':>6}")
print("-" * 36)
for m, a, n in zip(methods, aris, nmis):
    print(f"{m:<20} {a:>6.3f}  {n:>6.3f}")