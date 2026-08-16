import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

def sep(str):
    print(50 * '=' + str + 50 * '=')


iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

sep('raw data')
print(df.head())
print(df.info())
print(df.shape)

df = df.dropna()

sep('after droping nan')
print(df.isna().sum())
print(df.shape)

sep('heat map')
plt.figure(figsize = (16, 9))
sns.heatmap(df.corr(), cmap = 'coolwarm', center = 0, linewidths = .5)
plt.tight_layout()
plt.title('correlation heatmap')
plt.savefig('images/correlation_heatmap.png')

sns.pairplot(df, hue = 'target')
plt.title('pairplot')
plt.tight_layout()
plt.savefig('images/pairplot.png')

X = df.drop('target', axis = 1)
y = df['target']

sep('after separating x and y')
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

sep('after spliting')
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


model = RandomForestClassifier(n_estimators = 100, random_state = 42)

sep('train period')
print('train started')
model.fit(X_train, y_train)
print('train finished')
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)


sep('model scores')
print(f'Accuracy score: {accuracy_score(y_test, y_pred)}')
print(f'Precision score: {precision_score(y_test, y_pred, average = 'macro')}')
print(f'Recall Score: {recall_score(y_test, y_pred, average = 'macro')}')
print(f'F1 score: {f1_score(y_test, y_pred, average = 'macro')}')
print(f'ROC_AUC score: {roc_auc_score(y_test, y_prob, average = 'macro', multi_class = 'ovr')}')
print(f'Classification report: {classification_report(y_test, y_pred)}')

sep('plots')

# plt.figure(figsize = (16, 9))
plt.figure(figsize = (16, 9))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('confusion matrix')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig('images/confusion_matrix.png')

# RocCurveDisplay.from_predictions(y_test, y_prob)
# plt.title('roc curve')
# plt.tight_layout()
# plt.savefig('images/roc.png')

# PrecisionRecallDisplay.from_predictions(y_test, y_prob)
# plt.title('roc curve')
# plt.tight_layout()
# plt.savefig('images/precision.png')


df_importance = pd.DataFrame({'feature_name': X_train.columns, 'importance_score': model.feature_importances_})
df_importance = df_importance.sort_values(by = 'importance_score', ascending = False)
important_features = df_importance['feature_name'].head(3).tolist()
X_train_importance = X_train[important_features]
X_test_importance = X_test[important_features]

model_important = RandomForestClassifier(n_estimators = 100, random_state = 42)
print('important train started')
model_important.fit(X_train_importance, y_train)
print('important train finished')

y_pred_importance = model_important.predict(X_test_importance)
y_prob_importance = model_important.predict_proba(X_test_importance)

sep('model scores')
print(f'Accuracy score: {accuracy_score(y_test, y_pred_importance)}')
print(f'Precision score: {precision_score(y_test, y_pred_importance, average = 'macro')}')
print(f'Recall Score: {recall_score(y_test, y_pred_importance, average = 'macro')}')
print(f'F1 score: {f1_score(y_test, y_pred_importance, average = 'macro')}')
print(f'ROC_AUC score: {roc_auc_score(y_test, y_prob_importance, average = 'macro', multi_class = 'ovr')}')
print(f'Classification report: {classification_report(y_test, y_pred_importance)}')
