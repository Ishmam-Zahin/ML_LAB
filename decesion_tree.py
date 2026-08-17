import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('datasets/adult.csv')
print(df.head())
print(df.info())
print(df.shape)

df = df.replace('?', np.nan)

print(df.isna().sum())

df = df.dropna()

print(df.isna().sum())

for col in df.columns:
    if df[col].dtype == 'str':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])


plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), cmap = 'coolwarm', center = 0)
plt.tight_layout()
plt.savefig('images/corr.png')


X = df.drop('income', axis = 1)
y = df['income']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

model = DecisionTreeClassifier(random_state = 42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f'Accuracy scor: {accuracy_score(y_test, y_pred)}')
print(f'Report: {classification_report(y_test, y_pred)}')

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, cmap = 'Blues', annot = True)
plt.tight_layout()
plt.savefig('images/cm.png')