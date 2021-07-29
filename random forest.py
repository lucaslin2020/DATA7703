import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ssl
import collections
import csv
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier

all_data = pd.read_csv("D:/UQ/UQ_sem2_2020/DATA7703/Group Project/data_new_new/all_email.csv")

#print(all_data.shape)

X_train, X_test, y_train, y_test = train_test_split(all_data.iloc[:,:-1], all_data['label'], random_state=42)
#print(y_test)


#random forest
#n_estimators
accuracy_n = []
for i in range(10,101,10):
    random_forest = RandomForestClassifier(n_estimators=i,criterion='entropy',random_state=42)
    random_forest.fit(X_train, y_train)
    predictions_rf = random_forest.predict(X_test)
    accuracy_n.append(accuracy_score(y_test, predictions_rf))
print('random forest Max Accuracy score:', format(max(accuracy_n)))
plt.plot(range(10,101,10),accuracy_n)
plt.xlabel('n_estimators')
plt.ylabel('Accuracy')
plt.show()
#max_depth
accuracy_n = []
predictions_rf=0
for i in range(10,211,10):
    random_forest = RandomForestClassifier(n_estimators=50,max_depth=i,criterion='entropy',random_state=42)
    random_forest.fit(X_train, y_train)
    predictions_rf = random_forest.predict(X_test)
    print(accuracy_score(y_test, predictions_rf))
    accuracy_n.append(accuracy_score(y_test, predictions_rf))
print('random forest Max Accuracy score:', format(max(accuracy_n)))
plt.plot(range(10,211,10),accuracy_n)
plt.xlabel('n_estimators')
plt.ylabel('Accuracy')
plt.show()
#max_features
# accuracy_n = []
# predictions_rf=0
# for i in range(10,211,5):
#     random_forest = RandomForestClassifier(n_estimators=50,max_depth=140,max_features=i,criterion='entropy',random_state=42)
#     random_forest.fit(X_train, y_train)
#     predictions_rf = random_forest.predict(X_test)
#     print(accuracy_score(y_test, predictions_rf))
#     accuracy_n.append(accuracy_score(y_test, predictions_rf))
# print('random forest Max Accuracy score:', format(max(accuracy_n)))
# plt.plot(range(10,211,5),accuracy_n)
# plt.xlabel('n_estimators')
# plt.ylabel('Accuracy')
# plt.show()
#test
random_forest = RandomForestClassifier(n_estimators=50,max_depth=140,max_features=10,criterion='entropy',random_state=42)
random_forest.fit(X_train, y_train)
predictions_rf = random_forest.predict(X_test)
print('random forest Accuracy score:', format(accuracy_score(y_test, predictions_rf)))
print('random forest Precision score:', format(precision_score(y_test, predictions_rf)))
print('random forest Recall score:', format(recall_score(y_test, predictions_rf)))
print('random forest F1 score:', format(f1_score(y_test, predictions_rf)))

#10-fold CV
# X = all_data.iloc[:,:-1]
# y = all_data['label']
# X_train_list = []
# y_train_list = []
# X_test_list = []
# y_test_list = []
# kf = KFold(n_splits=10,shuffle=True,random_state=42)
# for train_index, test_index in kf.split(X):
#     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#     y_train, y_test = y.iloc[train_index], y.iloc[test_index]
#     X_train_list.append(X_train)
#     y_train_list.append(y_train)
#     X_test_list.append(X_test)
#     y_test_list.append(y_test)
# accuracy_sum = 0
# for i in range(len(X_train_list)):
#     random_forest = RandomForestClassifier(n_estimators=50,max_depth=140,max_features=10,criterion='entropy',random_state=42)
#     random_forest.fit(X_train_list[i], y_train_list[i])
#     predictions_rf = random_forest.predict(X_test_list[i])
#     accuracy_sum += accuracy_score(y_test_list[i], predictions_rf)
# average_accuracy = accuracy_sum/10
# print(average_accuracy) #0.9288811024595229
#
# #knn
# k_neighbor = KNeighborsClassifier(n_neighbors=2,weights='uniform')
# k_neighbor.fit(X_train, y_train)
# predictions_knn = k_neighbor.predict(X_test)
# print('knn Accuracy score:', format(accuracy_score(y_test, predictions_knn)))
# print('knn Precision score:', format(precision_score(y_test, predictions_knn)))
# print('knn Recall score:', format(recall_score(y_test, predictions_knn)))
# print('knn F1 score:', format(f1_score(y_test, predictions_knn)))


