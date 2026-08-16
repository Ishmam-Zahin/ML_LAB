import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    RocCurveDisplay, PrecisionRecallDisplay
)

def sep(str = 'NaN'):
    print(60 * '=' + str + 60 * '=')


df = pd.read_csv('/home/zahin/Desktop/ML_LAB/datasets/adult.csv')

sep('raw data inspection')
print(df.head())
print(df.info())
print(df.shape)
print(df.isna().sum())


df = df.replace('?', np.nan)

sep('replace ? with nan')
print(df.isnull().sum())
print(df.shape)

df = df.drop(['fnlwgt', 'education'], axis=1)
df = df.dropna()

sep('after dropping nan')
print(df.isna().sum())
print(df.info())
print(df.shape)

le_income = LabelEncoder()
df['income'] = le_income.fit_transform(df['income'])
nominal_cols = [col for col in df.columns if df[col].dtype == 'str' and col != 'income']
df = pd.get_dummies(df, columns = nominal_cols, drop_first = True)

sep('after applying encoders')
print(df.shape)
print(df.info())
print(df.head())


X = df.drop('income', axis = 1)
y = df['income']

sep('after separating features and target columns')
print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

sep('after train test split')

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = LogisticRegression(max_iter = 1000)
model.fit(X_train_scaled, y_train)

model2 = LogisticRegression(penalty = 'l2', C = 1.0, max_iter = 1000)
model2.fit(X_train_scaled, y_train)

model1 = LogisticRegression(penalty = 'l1', solver = 'liblinear', C = 1.0, max_iter = 1000)
model1.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

y_pred1 = model1.predict(X_test_scaled)
y_prob1 = model1.predict_proba(X_test_scaled)[:, 1]

y_pred2 = model2.predict(X_test_scaled)
y_prob2 = model2.predict_proba(X_test_scaled)[:, 1]

sep('model with no regularization scores')

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print(classification_report(y_test, y_pred))

sep('model with l1 regularization scores')

print("Accuracy:", accuracy_score(y_test, y_pred1))
print("Precision:", precision_score(y_test, y_pred1))
print("Recall:", recall_score(y_test, y_pred1))
print("F1:", f1_score(y_test, y_pred1))
print("ROC-AUC:", roc_auc_score(y_test, y_prob1))
print(classification_report(y_test, y_pred1))

sep('model with l2 regularization scores')

print("Accuracy:", accuracy_score(y_test, y_pred2))
print("Precision:", precision_score(y_test, y_pred2))
print("Recall:", recall_score(y_test, y_pred2))
print("F1:", f1_score(y_test, y_pred2))
print("ROC-AUC:", roc_auc_score(y_test, y_prob2))
print(classification_report(y_test, y_pred2))


sep('mode with no regularization plots')

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig(f'images/Confusion Matrix.png')

RocCurveDisplay.from_predictions(y_test, y_prob)
plt.title("ROC Curve")
plt.savefig(f'images/ROC_Curve.png')

PrecisionRecallDisplay.from_predictions(y_test, y_prob)
plt.title("Precision-Recall Curve")
plt.savefig(f'images/Precision-Recall_Curve.png')

coeffs = pd.Series(model.coef_[0], index=X.columns).sort_values()
coeffs.tail(15).plot(kind='barh')
plt.title("Top 15 Positive Coefficients")
plt.savefig(f'images/Top_15_Positive_Coefficients.png')

sep('mode with l1 regularization plots')

cm1 = confusion_matrix(y_test, y_pred1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig(f'images/Confusion_Matrix_l1.png')

RocCurveDisplay.from_predictions(y_test, y_prob1)
plt.title("ROC Curve")
plt.savefig(f'images/ROC_Curve_l1.png')

PrecisionRecallDisplay.from_predictions(y_test, y_prob1)
plt.title("Precision-Recall Curve")
plt.savefig(f'images/Precision-Recall_Curve_l1.png')

coeffs1 = pd.Series(model.coef_[0], index=X.columns).sort_values()
coeffs1.tail(15).plot(kind='barh')
plt.title("Top 15 Positive Coefficients")
plt.savefig(f'images/Top_15_Positive_Coefficients_l1.png')

sep('mode with l2 regularization plots')

cm2 = confusion_matrix(y_test, y_pred2)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig(f'images/Confusion_Matrix_l2.png')

RocCurveDisplay.from_predictions(y_test, y_prob2)
plt.title("ROC Curve")
plt.savefig(f'images/ROC_Curve_l2.png')

PrecisionRecallDisplay.from_predictions(y_test, y_prob2)
plt.title("Precision-Recall Curve")
plt.savefig(f'images/Precision-Recall_Curve_l2.png')

coeffs2 = pd.Series(model.coef_[0], index=X.columns).sort_values()
coeffs2.tail(15).plot(kind='barh')
plt.title("Top 15 Positive Coefficients")
plt.savefig(f'images/Top_15_Positive_Coefficients_l2.png')