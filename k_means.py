import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


X, y = make_blobs(n_samples = 1000, centers = 4, random_state = 42)

plt.figure(figsize = (4, 4))
plt.scatter(x = X[:, 0], y = X[:, 1], c = y)
plt.tight_layout()
plt.savefig('images/dataset.png')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

from sklearn.cluster import KMeans
model = KMeans(n_clusters = 4, random_state = 42)
model.fit(X_train)
y_pred = model.predict(X_test)
ari = adjusted_rand_score(y_test, y_pred)
nmi = normalized_mutual_info_score(y_test, y_pred)
print(ari)
print(nmi)

model_full = KMeans(n_clusters = 4, random_state = 42)
model_full.fit(X)
centers = model_full.cluster_centers_
y_pred_full = model_full.labels_


plt.figure(figsize=(12, 8))
plt.scatter(X[:, 0], X[:, 1], c = y_pred_full)
plt.scatter(centers[:, 0], centers[:, 1], marker= 'x')
plt.tight_layout()
plt.savefig('images/center.png')

ks = []
for i in range(11):
    i = i + 1
    model_tmp = KMeans(n_clusters = i, random_state = 42)
    model_tmp.fit(X)
    ks.append(model_tmp.inertia_)


plt.figure(figsize = (12, 8))
plt.plot(range(11), ks)
plt.tight_layout()
plt.savefig('images/elbow.png')