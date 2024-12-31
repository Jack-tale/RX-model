#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.model_selection import StratifiedShuffleSplit, RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
import sklearn.metrics as sm
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn import neighbors


# ## 1. SVM

pickle_file = './Training data/data.pickle'
with open(pickle_file, 'rb') as f:
    pickle_data = pickle.load(f)
    X_train = pickle_data['train_dataset']
    y_train = pickle_data['train_labels']
    X_test = pickle_data['test_dataset']
    y_test = pickle_data['test_labels']
    del pickle_data

print('The data has been loaded')

svm_model = SVC(probability=True)
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)
score = accuracy_score(y_pred, y_test)
print(score)


# 调参

Cs = [0.01, 0.1, 1, 10, 100]
gammas = [0.01, 0.1, 1, 10, 100]

svm = Pipeline([('scaler', StandardScaler()),('svc', OneVsRestClassifier(SVC()))])
params_grid = dict(svc__estimator__C = Cs, svc__estimator__gamma = gammas)
cv = StratifiedShuffleSplit(n_splits = 10, test_size = .2, random_state = 1)
grid_cv = RandomizedSearchCV(svm, param_distributions = params_grid, cv = cv, n_iter = 10, n_jobs = -1)
grid_cv.fit(X_train, y_train)
print('Best params:', grid_cv.best_params_)
print('Best score: ', grid_cv.best_score_)

list(zip(grid_cv.cv_results_['params'],grid_cv.cv_results_['mean_test_score']))



Cs = [1, 10, 100, 500]
gammas = [0.01, 0.05,0.1, 0.5]

svm = Pipeline([('scaler', StandardScaler()),('svc', OneVsRestClassifier(SVC()))])
params_grid = dict(svc__estimator__C = Cs, svc__estimator__gamma = gammas)
cv = StratifiedShuffleSplit(n_splits = 5, test_size = .2, random_state = 1)
grid_cv = GridSearchCV(svm, param_grid = params_grid, cv = cv, n_jobs = -1)
grid_cv.fit(X_train, y_train)
print('Best params:', grid_cv.best_params_)
print('Best score: ', grid_cv.best_score_)


y_pred = grid_cv.predict(X_test)
con_matrix = sm.confusion_matrix(y_test, y_pred, labels=['Coal', 'Construction', 'Road dust', 'Soil', 'Steel', 'Biomass'])
print(con_matrix)
report = sm.classification_report(y_test, y_pred)
print('分类报告为：', report , sep='\n')


y_pred = grid_cv.predict(X_test)
score = accuracy_score(y_pred, y_test)
print(score)



# ## 2. RF

classifier = RandomForestClassifier(n_estimators=50, criterion='gini', random_state=42)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
accuracy_score(y_test, y_pred)



score_lt = []

for i in range(0,200,10):
    rfc = RandomForestClassifier(n_estimators=i+1
                                ,random_state=90)
    score = cross_val_score(rfc, X_train, y_train, cv=5).mean()
    score_lt.append(score)
score_max = max(score_lt)
print('最大得分：{}'.format(score_max),
      '子树数量为：{}'.format(score_lt.index(score_max)*10+1))


x = np.arange(1,201,10)
plt.subplot(111)
plt.plot(x, score_lt, 'r-')
plt.show()


rfc = RandomForestClassifier(n_estimators=100, random_state=90)

# 用网格搜索调整max_depth
param_grid = {'max_depth':np.arange(1,20)}
GS = GridSearchCV(rfc, param_grid, cv=10)
GS.fit(X_train, y_train)

best_param = GS.best_params_
best_score = GS.best_score_
print(best_param, best_score)


GS.best_estimator_


y_pred = GS.predict(X_test)
print(accuracy_score(y_test, y_pred))



# ## KNN


clf = neighbors.KNeighborsClassifier(n_neighbors = 15, weights = 'distance')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('Acurancy score:',accuracy_score(y_test, y_pred))


knn_est = Pipeline([('scaler', StandardScaler()), ('knn',neighbors.KNeighborsClassifier())])

params_grid = dict(knn__n_neighbors = [10, 20, 30, 50, 80], 
                   knn__p = [1, 2], 
                   knn__weights = ['uniform', 'distance']
                  )
cv = StratifiedShuffleSplit(n_splits = 10, test_size = .2, random_state = 1)

grid_cv = RandomizedSearchCV(knn_est, param_distributions = params_grid, cv = cv, n_iter = 10)
grid_cv.fit(X_train, y_train)
print('Model training ends.')
grid_cv.best_score_
grid_cv.best_params_


grid_cv.best_score_


knn_est = Pipeline([('scaler', StandardScaler()), ('knn',neighbors.KNeighborsClassifier(weights = 'distance'))])

params_grid = dict(knn__n_neighbors = [1, 5, 10, 20], knn__p = [1, 2])
cv = StratifiedShuffleSplit(n_splits = 10, test_size = .2, random_state = 1)

grid_cv = GridSearchCV(knn_est, param_grid = params_grid, cv = cv)
grid_cv.fit(X_train, y_train)
print('Model training ends.')
print(grid_cv.best_score_)
print(grid_cv.best_params_)


y_pred = grid_cv.predict(X_test)
accuracy_score(y_test,y_pred)
