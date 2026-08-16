import numpy as np
import matplotlib as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report,
    roc_auc_score, RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.neighbors import KNeighborsClassifier

def sep(str):
    print(50 * '=' + str + 50 * '=')

df = pd.read_csv('datasets/adult.csv')

sep('raw data')
print(df.head())
print(df.info())
print(df.shape)

df = df.replace('?', np.nan)

sep('afer replacing ? with nan')
print(df.isna().sum())
print(df.shape)

df = df.dropna()

sep('after droping nan')
print(df.isna().sum())
print(df.shape)

le_income = LabelEncoder()
df['income'] = le_income.fit_transform(df['income'])
nominal_cols = [col for col in df.columns if df[col].dtype == 'str' and col != 'income']
df = pd.get_dummies(df, columns = nominal_cols, drop_first = True)

sep('after encoding')
print(df.head())
print(df.shape)

X = df.drop('income', axis = 1)
y = df['income']

sep('after separating x and y')
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

sep('after spliting')
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

accuracys = []

for k in range(1, 21):
    print(f'starting for {k}')
    model = KNeighborsClassifier(n_neighbors = k)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    score = accuracy_score(y_test, y_pred)
    accuracys.append(score)

best_k = np.argmax(accuracys) + 1

model = KNeighborsClassifier(n_neighbors = best_k)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]


sep('model scores')
print(f'Accuracy score: {accuracy_score(y_test, y_pred)}')
print(f'Precision score: {precision_score(y_test, y_pred)}')
print(f'Recall Score: {recall_score(y_test, y_pred)}')
print(f'F1 score: {f1_score(y_test, y_pred)}')
print(f'ROC_AUC score: {roc_auc_score(y_test, y_prob)}')
print(f'Classification report: {classification_report(y_test, y_pred)}')