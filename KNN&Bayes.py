#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 19:48:01 2022

@author: rajsinghbani
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as sk_m
import scipy.stats as sps
data = pd.read_csv("/Users/rajsinghbani/Downloads/LAB4-2022/SteelPlateFaults-2class.csv")
[X_train, X_test, X_label_train, X_label_test] = train_test_split(data, data.Class, test_size=0.3, random_state=42, shuffle=True)
cla2 = X_test.groupby('Class')
zero = cla2.get_group(0)
one = cla2.get_group(1)
for i in range(1,6,2):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train.iloc[:,:27],X_train.Class)
    y_pred = knn.predict(X_test.iloc[:,:27])
    y_true = X_label_test.tolist()
    print(sk_m.confusion_matrix(y_true, y_pred))
    print()
    print(sk_m.accuracy_score(y_true, y_pred))
    print()
X_train.to_csv('SteelPlateFaults-train.csv', index=False)
X_test.to_csv('SteelPlateFaults-test.csv', index=False)

#2

X_train_1 = pd.read_csv("/Users/rajsinghbani/Documents/SteelPlateFaults-train.csv")
X_test_1 = pd.read_csv("/Users/rajsinghbani/Documents/SteelPlateFaults-test.csv")
X_train_1.drop('Class',axis=1,inplace=True)
X_test_1.drop('Class',axis=1,inplace=True)
for i in X_train_1.columns:
    maxi = X_train_1[i].max()
    mini = X_train_1[i].min()
    X_train_1[i]=(X_train_1[i]-mini)/(maxi-mini)
    X_test_1[i]=(X_test_1[i]-mini)/(maxi-mini)
X_train.to_csv('SteelPlateFaults-train-Normalised.csv', index=False)
X_test.to_csv('SteelPlateFaults-test-Normalised.csv', index=False)
for i in range(1,6,2):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train_1,X_label_train)
    y_pred = knn.predict(X_test_1)
    y_true = X_label_test.tolist()
    print(sk_m.confusion_matrix(y_true, y_pred))
    print()
    print(sk_m.accuracy_score(y_true, y_pred))
    print()
    
#3

y_true = X_label_test.tolist()
X_train_2 = pd.read_csv("/Users/rajsinghbani/Documents/SteelPlateFaults-train.csv")
X_test_2 = pd.read_csv("/Users/rajsinghbani/Documents/SteelPlateFaults-test.csv")
group_0 = X_train_2.groupby('Class')
class_0 = group_0.get_group(0).iloc[:,:27]
class_1 = group_0.get_group(1).iloc[:,:27]
C_0 = class_0.shape[0]
C_1 = class_1.shape[0]
P_C_0 = C_0/(C_0 + C_1)
P_C_1 = C_1/(C_0 + C_1)
cov_mat_0 = np.cov(class_0.T)
cov_mat_1 = np.cov(class_1.T)
mean_0 = np.array(class_0.mean())
mean_1 = np.array(class_1.mean())
def mul_pdf(x,u,cov):
    d = x.shape[0]/2
    det = abs(np.linalg.det(cov))
    tr = np.transpose(x-u)
    inve = np.linalg.inv(cov)
    pdf = 1/(((2*np.pi)**d)*det**(1/2))
    pdf *= np.exp((-1/2)*np.matmul(np.matmul(tr,inve),(x-u)))
    return pdf
pred_bayes = []
X_test_2.drop('Class',axis=1,inplace=True)
for i in range(X_test_2.shape[0]):
    prob_0 = mul_pdf(np.array(X_test_2.iloc[i]),mean_0,cov_mat_0)*P_C_0
    prob_1 = mul_pdf(np.array(X_test_2.iloc[i]),mean_1,cov_mat_1)*P_C_1
    if prob_0>prob_1:
        pred_bayes.append(0)
    else:
        pred_bayes.append(1)

print('bayes')
print()
print(sk_m.confusion_matrix(y_true, pred_bayes))
print()
print(sk_m.accuracy_score(y_true, pred_bayes))
print()

#4

