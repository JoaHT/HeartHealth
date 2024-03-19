#This project will focus on Heart health, what connections we can draw from excisting information 
#and if they have an impact on heart attacks through machine learning algorhitms. As well as 
#making an informative dashboard on the information we will acquire. 

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

data = pd.read_csv('Heart_health.csv')

data.head()

#During data cleaning we are checking if theres any null values or duplicates
#and if there are we remove them
data.isna().sum()

data.duplicated().sum()

duplicated = data[data.duplicated()]
duplicated

data = data.drop_duplicates()

#Here we can see that the dataset is still intact after removing duplicates
data.head()

data.shape

data.info()

data['Blood Pressure(mmHg)'].unique()
data['Smoker'].unique()
data['Heart Attack'].unique()
data['Gender'].unique()

data_new = data.drop(['ID'], axis=1)
data_new.duplicated().sum()
data_new = data_new.drop_duplicates()

data_new.to_csv('cleaned_heart.csv', sep=',',index=False,encoding='utf-8')

#Here we change Yes and No to 1 and 0, to make it easier to plot into the MLA
data_new['Smoker'] = data_new['Smoker'].map({'Yes':1, 'No':0})

data_new['Smoker'].unique()

data_new.info()

data_new[['Systolic Blood Pressure','Diastolic Blood Pressure']] = data_new['Blood Pressure(mmHg)'].str.split('/',n=1, expand=True)
data_new.head()

data_mla = data_new.drop(['Name','Blood Pressure(mmHg)'], axis=1)

data_mla = pd.get_dummies(data_mla, columns=['Gender'], drop_first=True)
data_mla.head()

y = data_mla['Heart Attack']
x = data_mla.copy()
x = data_mla.drop('Heart Attack', axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.25, stratify=y, random_state=5)

cv_params = {'max_depth':[2,3,4,5],
             'min_samples_leaf':[3,4,5],
             'min_samples_split':[2,3,4],
             'max_features':[3,4,5],
             'n_estimators':[50,75,100]}

rf = RandomForestClassifier(random_state=0)

scoring = ['precision', 'f1', 'recall', 'accuracy']

rf_cv = GridSearchCV(rf, cv_params, scoring=scoring, cv=5, refit='recall')

rf_cv.fit(X_train, Y_train)

rf_cv.best_params_

rf_cv.best_estimator_

rf_cv.best_score_

y_pred = rf_cv.predict(X_test)

recall = recall_score(Y_test, y_pred)
accuracy = accuracy_score(Y_test, y_pred)
precision = precision_score(Y_test, y_pred)
f1 = f1_score(Y_test, y_pred)

print('Recall:', recall,'\n'
      'Accuracy:', accuracy,'\n'
      'Precision:', precision,'\n'
      'F1', f1,'\n')

cm = confusion_matrix(Y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()

feature_importance = pd.Series(rf_cv.best_estimator_.feature_importances_, index=X_train.columns)
feature_importance.sort_values(ascending=True,inplace=True)
feature_importance.plot.barh(color='green')
plt.xlabel('Importance')
plt.ylabel("Feature")
plt.title('Heart Attack Feature Importance')
