import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split


X_nolap, y_nolap = make_blobs(n_samples = 1000, centers = 2, cluster_std = 1.2, random_state = 42)
X_lap, y_lap = make_blobs(n_samples = 1000, centers = 4, cluster_std = 3.0, random_state = 42)
df_nolap = pd.DataFrame({'feature1': X_nolap[:, 0], 'feature2': X_nolap[:, 1], 'target': y_nolap})
df_lap = pd.DataFrame({'feature1': X_lap[:, 0], 'feature2': X_lap[:, 1], 'target': y_lap})

plt.figure(figsize = (16, 9))
plt.subplot(1, 2, 1)
sns.scatterplot(data = df_nolap, x = 'feature1', y = 'feature2', hue = 'target')
plt.title('non overlapping binary data')

plt.subplot(1, 2, 2)
sns.scatterplot(data = df_lap, x = 'feature1', y = 'feature2', hue = 'target')
plt.title('overlapping multiclass data')
plt.tight_layout()
plt.savefig('images/dataset.png')


X = df_nolap.drop('target', axis = 1)
y = df_nolap['target']

X_train_nolap, X_test_nolap, y_train_nolap, y_test_nolap = train_test_split(X, y, test_size = 0.2, random_state = 42, shuffle = True)

X = df_lap.drop('target', axis = 1)
y = df_lap['target']

X_train_lap, X_test_lap, y_train_lap, y_test_lap = train_test_split(X, y, test_size = 0.2, random_state = 42, shuffle = True)

model_nolap = SVC(kernel = 'linear', C = 0.1)
model_nolap.fit(X_train_nolap, y_train_nolap)
y_pred_nolap = model_nolap.predict(X_test_nolap)
score_nolap = accuracy_score(y_test_nolap, y_pred_nolap)
print(score_nolap)

model_lap = SVC(kernel = 'rbf', C = .1)
model_lap.fit(X_train_lap, y_train_lap)
y_pred_lap = model_lap.predict(X_test_lap)
score_lap = accuracy_score(y_test_lap, y_pred_lap)
print(score_lap)

X_arr_nolap = X_train_nolap.values
X_arr_lap = X_train_lap.values

x_min_nolap, x_max_nolap = X_arr_nolap[:, 0].min() - 1, X_arr_nolap[:, 0].max() + 1
y_min_nolap, y_max_nolap = X_arr_nolap[:, 1].min() - 1, X_arr_nolap[:, 1].max() + 1

x_min_lap, x_max_lap = X_arr_lap[:, 0].min() - 1, X_arr_lap[:, 0].max() + 1
y_min_lap, y_max_lap = X_arr_lap[:, 1].min() - 1, X_arr_lap[:, 1].max() + 1

xx_nolap, yy_nolap = np.meshgrid(
    np.linspace(x_min_nolap, x_max_nolap, 300),
    np.linspace(y_min_nolap, y_max_nolap, 300),
)

xx_lap, yy_lap = np.meshgrid(
    np.linspace(x_min_lap, x_max_lap, 300),
    np.linspace(y_min_lap, y_max_lap, 300),
)

grid_nolap = pd.DataFrame(np.c_[xx_nolap.ravel(), yy_nolap.ravel()], columns = ['feature1', 'feature2'])
grid_lap = pd.DataFrame(np.c_[xx_lap.ravel(), yy_lap.ravel()], columns = ['feature1', 'feature2'])

z_nolap = model_nolap.decision_function(grid_nolap).reshape(xx_nolap.shape)
z_lap = model_lap.predict(grid_lap).reshape(xx_lap.shape)

plt.figure(figsize = (16, 9))
plt.subplot(1, 2, 1)
plt.scatter(
    X_arr_nolap[:, 0], X_arr_nolap[:, 1],
    c = y_train_nolap, cmap = 'coolwarm',
    edgecolors = 'k', s = 25
)
plt.contour(xx_nolap, yy_nolap, z_nolap, colors='k', levels=[-1, 0, 1],
    linestyles=['--', '-', '--'],
    linewidths=[1, 1.5, 1])

plt.scatter(
    model_nolap.support_vectors_[:, 0],
    model_nolap.support_vectors_[:, 1],
    s=140, facecolors='none', edgecolors='green',
    linewidths=1.8,
    label=f'Support Vectors ({len(model_nolap.support_vectors_)})'
)

plt.subplot(1, 2, 2)
plt.scatter(
    X_arr_lap[:, 0], X_arr_lap[:, 1],
    c = y_train_lap, cmap = 'coolwarm',
    edgecolors = 'k', s = 25
)
plt.contourf(xx_lap, yy_lap, z_lap, alpha=0.3, cmap='tab10', fmt='d')

plt.scatter(
    model_lap.support_vectors_[:, 0],
    model_lap.support_vectors_[:, 1],
    s=140, facecolors='none', edgecolors='green',
    linewidths=1.8,
    label=f'Support Vectors ({len(model_nolap.support_vectors_)})'
)

plt.tight_layout()
plt.savefig('images/dec_nolap.png')
