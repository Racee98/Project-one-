#!/usr/bin/env python



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
from sklearn import metrics

df = pd.read_csv('diabetes-dataset.csv')
df = df.rename(columns={'Outcome': 'DiabetesOutcome'})
df.describe()

plt.figure(figsize=(9,7))
sns.countplot(x='DiabetesOutcome', hue='DiabetesOutcome', data=df).set(title='Diabetes cases', ylabel='Count of Cases', xlabel='Diabetes Status')
plt.show()

plt.figure(figsize=(9,7))
sns.histplot(x='BMI', data=df).set(title='A Histogram of BMI', ylabel='Count', xlabel='BMI')
plt.show()

plt.figure(figsize=(9,7))
sns.histplot(x='Age', data=df).set(title='Age Distribution', ylabel='Count', xlabel='Age')
plt.show()

plt.figure(figsize=(9,7))
sns.histplot(x='Pregnancies', binwidth=1, data=df).set(title='Distribution of Pregnancies Count', ylabel='Count', xlabel='Pregnancies')
plt.show()

corr = df.corr()
plt.figure(figsize=(9,7))
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, cmap="Greens", annot=True).set(title='Correlation Plot')
plt.show()

plt.figure(figsize=(9,7))
sns.scatterplot(df, y='BloodPressure', x='Age', hue='DiabetesOutcome').set(title='Blood Pressure vs Age', ylabel='Blood Pressure', xlabel='Age')
plt.show()

plt.figure(figsize=(9,7))
sns.scatterplot(df, y='Glucose', x='Age', hue='DiabetesOutcome').set(title='Glucose vs Age', ylabel='Glucose', xlabel='Age')
plt.show()

plt.figure(figsize=(9,7))
sns.scatterplot(df, y='SkinThickness', x='Insulin', hue='DiabetesOutcome').set(title='Skin Thickness vs Insulin', ylabel='Skin Thickness', xlabel='Insulin')
plt.show()

df_train = df.sample(round(len(df)*0.8))
df_test = df.drop(df_train.index)

formula = 'DiabetesOutcome ~ Pregnancies+Glucose+BloodPressure+SkinThickness+Insulin+BMI+DiabetesPedigreeFunction+Age'

model = smf.glm(formula=formula, data=df_train, family=sm.families.Poisson())
result = model.fit()

result = result.predict(df_test)
result[result > 0.5] = 1
result[result <= 0.5] = 0

confusion_matrix = metrics.confusion_matrix(df_test.DiabetesOutcome, result)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()

accuracy = metrics.accuracy_score(df_test.DiabetesOutcome, result)
print(accuracy)
