import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler


X_blob, y_blob = make_blobs(n_samples = 1000, centers = 2, cluster_std = 1.5, random_state = 42)
X_circle, y_circle = make_circles(n_samples = 1000, noise = 0.05, factor = 0.4, random_state = 42)
X_moon, y_moon = make_moons(n_samples = 1000, noise = 0.1, random_state = 42)

plt.figure(figsize = (16, 9))
plt.subplot(1, 3, 1)
plt.scatter(X_blob[:, 0], X_blob[:, 1], c = y_blob)
plt.title('blob')
plt.subplot(1, 3, 2)
plt.scatter(X_moon[:, 0], X_moon[:, 1], c = y_moon)
plt.title('moon')
plt.subplot(1, 3, 3)
plt.scatter(X_circle[:, 0], X_circle[:, 1], c = y_circle)
plt.title('circle')

plt.tight_layout()
plt.savefig('images/dataset.png')

scaler = StandardScaler()
X_circle = scaler.fit_transform(X_circle)

model_km = KMeans(n_clusters = 2, random_state = 42)
model_hr = AgglomerativeClustering(n_clusters = 2)
model_db = DBSCAN(eps = 0.4, min_samples = 5)
model_gm = GaussianMixture(n_components = 2, n_init = 10)

model_km.fit(X_blob, y_blob)
model_hr.fit(X_moon, y_moon)
model_db.fit(X_circle, y_circle)
model_gm.fit(X_blob, y_blob)


plt.figure(figsize = (16, 9))
plt.subplot(2, 2, 1)
plt.scatter(X_blob[:, 0], X_blob[:, 1], c = model_km.labels_)
plt.scatter(model_km.cluster_centers_[:, 0], model_km.cluster_centers_[:, 1], marker = 'x')
plt.title('K means')

plt.subplot(2, 2, 2)
plt.scatter(X_moon[:, 0], X_moon[:, 1], c = model_hr.labels_)
plt.title('Hiararchi')

plt.subplot(2, 2, 3)
plt.scatter(X_circle[:, 0], X_circle[:, 1], c = model_db.labels_)
plt.title('DBscan')


plt.subplot(2, 2, 4)
plt.scatter(X_blob[:, 0], X_blob[:, 1], c = model_gm.predict(X_blob))
plt.title('gmm')

plt.tight_layout()
plt.savefig('images/dec.png')


X_par = X_moon[:100]
z = linkage(X_par, method = 'ward')

plt.figure(figsize=(12, 8))
dendrogram(z)

plt.tight_layout()
plt.savefig('images/dendo.png')