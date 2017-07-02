#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 18:20:46 2017

@author: syedrahman
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt 
plt.style.use('ggplot')
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Imputer
import itertools as it
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

os.chdir('/Users/syedrahman/Documents/Summer2017/Insight/breast-cancer-wisconsin')

bcw = pd.read_csv('breast-cancer-wisconsin.txt',header=None)

### dropping 

#columns = ['Sample code number',
'Clump Thickness',
'Uniformity of Cell Size',
'Uniformity of Cell Shape',
'Marginal Adhesion',
'Single Epithelial Cell Size',
'Bare Nuclei',
'Bland Chromatin',
'Normal Nucleoli',
'Mitoses', 
'Class']
#bcw.columns = [x.lower().replace(' ', '') for x in columns]


### convert to numeric
bcw = bcw.apply(pd.to_numeric,errors='coerce')

### checking if there are any null values
pd.isnull(bcw).apply(sum, axis = 0)

### doing a scatter plot to see if bcw actually matter. We first try dropping the na cases
#plt.scatter(bcw[6],bcw[10])

bcw_drop = bcw.dropna(axis=0)

Xcolumns = bcw_drop.columns[1:10]
ycolumns = bcw_drop.columns[10]
X = bcw_drop[Xcolumns].as_matrix()

### recoding malignant tumors as 1 
y = (bcw_drop[ycolumns]==4)*1.0

maxf1 = 0
for i in range(1,10):
    varCombs = list(it.combinations(range(0,9),i))
    for j in range(len(varCombs)):
        Xmat = X[:,varCombs[j]]
        X_train, X_test, y_train, y_test = train_test_split(
                Xmat, y, test_size=0.3, random_state=1)
        clf = RandomForestClassifier(n_jobs=2)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if(f1_score(y_test,y_pred)>maxf1):
            maxf1=f1_score(y_test,y_pred)
            bestmodel = varCombs[j]
            bestaccuracy = accuracy_score(y_test,y_pred)
            bestrecall = recall_score(y_test,y_pred)
            bestprecision = precision_score(y_test,y_pred)




imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(bcw)
bcw_imp = imp.transform(bcw)
bcw_imp.shape
X = bcw_imp[:,1:10]
y = bcw_imp[:,10]
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1)

clf = RandomForestClassifier(n_jobs=2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_predpa = clf.predict_proba(X_test)
confusion_matrix(y_test,y_pred)




df = pd.concat([X,y], axis = 1)
df.head()








sns_plot = sns.pairplot(df, hue="class")
sns_plot.savefig("output.png")






x == y 
for x in Xcolumns:
    for y in Xcolumns:
        if ~(x==y):
            X[x+y] = X[x]*X[y]
