import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = make_blobs(n_samples = 1000, centers = 3, cluster_std = 2.2, random_state = 42)

plt.figure(figsize = (12, 8))
plt.scatter(X[:, 0], X[:, 1], c = y)
plt.savefig('images/dataset.png')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVC(C = 0.1, kernel = 'rbf')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

score = accuracy_score(y_test, y_pred)
print(score)

x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

grid = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns = ['feature1', 'feature2'])

z = model.predict(grid).reshape(xx.shape)

plt.figure(figsize = (12, 8))
plt.scatter(X_train[:, 0], X_train[:, 1], c = y_train)
plt.contourf(xx, yy, z, alpha=0.3, cmap='tab10')
plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], marker = 'x')
plt.savefig('images/dec.png')
