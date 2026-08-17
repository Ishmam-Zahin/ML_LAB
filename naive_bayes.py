import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


df = pd.read_csv('/home/zahin/Desktop/ML_LAB/datasets/diabetes.csv')
print(df.head())
print(df.info())
print(df.shape)

cols_with_missing = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_missing] = df[cols_with_missing].replace(0, np.nan)

df = df.dropna()

print(df.isna().sum())

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), cmap = 'coolwarm', center = 0)
plt.tight_layout()
plt.savefig('images/corr.png')

X = df.drop('Outcome', axis = 1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, stratify= y, random_state = 42)

model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f'Accuracy scor: {accuracy_score(y_test, y_pred)}')
print(f'Report: {classification_report(y_test, y_pred)}')

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, cmap = 'Blues', annot = True)
plt.tight_layout()
plt.savefig('images/cm.png')