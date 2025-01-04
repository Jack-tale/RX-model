import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import datetime

from sklearn.svm import SVC
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, make_scorer, roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, GridSearchCV, cross_val_score, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, label_binarize
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
import gc


pickle_file = '../Row data/train_test_data_1.pickle'
with open(pickle_file, 'rb') as f:
    pickle_data = pickle.load(f)
    X_train = pickle_data['train_dataset']
    y_train = pickle_data['train_labels']
    X_test = pickle_data['test_dataset']
    y_test = pickle_data['test_labels']
    del pickle_data
    gc.collect()
print('The data has been loaded')


# ## 1. SVM


# Simple test

svm_model = SVC(probability=True)
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)
score = accuracy_score(y_pred, y_test)
print(score)


# Parameter_tuning_1

Cs = [0.01, 0.1, 1, 10, 100]
gammas = [0.01, 0.1, 1, 10, 100]

svm = Pipeline([('scaler', StandardScaler()),('svc', OneVsRestClassifier(SVC(probability=True)))])
params_grid = dict(svc__estimator__C = Cs, svc__estimator__gamma = gammas)
cv = StratifiedKFold(n_splits = 5, shuffle = True)
SVM_model = RandomizedSearchCV(svm, param_distributions = params_grid, cv = cv, n_iter = 16, n_jobs = -1)

starttime = datetime.datetime.now()
SVM_model.fit(X_train, y_train)
endtime = datetime.datetime.now()

print('Run time:', endtime - starttime)
print(list(zip(SVM_model.cv_results_['params'],SVM_model.cv_results_['mean_test_score'])))
print('Best params:', SVM_model.best_params_)
print('Best score: ', SVM_model.best_score_)


# Parameter_tuning_2

Cs = [1, 10, 100, 500]
gammas = [0.01, 0.05,0.1, 0.5]

svm = Pipeline([('scaler', StandardScaler()),('svc', OneVsRestClassifier(SVC()))])
params_grid = dict(svc__estimator__C = Cs, svc__estimator__gamma = gammas)
cv = StratifiedKFold(n_splits = 5, shuffle = True)
SVM_model = GridSearchCV(svm, param_grid = params_grid, cv = cv, n_jobs = -1)

starttime = datetime.datetime.now()
SVM_model.fit(X_train, y_train)
endtime = datetime.datetime.now()

print('Run time:', endtime - starttime)
print(list(zip(SVM_model.cv_results_['params'],SVM_model.cv_results_['mean_test_score'])))
print('Best params:', SVM_model.best_params_)
print('Best score: ', SVM_model.best_score_)


# Parameter_tuning_3

Cs = [10, 100, 500, 1000]
gammas = [0.01, 0.1, 0.5, 1]

svm = Pipeline([('scaler', StandardScaler()),('svc', OneVsRestClassifier(SVC(probability=True)))])
params_grid = dict(svc__estimator__C = Cs, svc__estimator__gamma = gammas)
cv = StratifiedKFold(n_splits = 5, shuffle = True)
SVM_model = RandomizedSearchCV(svm, param_distributions = params_grid, cv = cv, n_iter = 10, n_jobs = -1)

starttime = datetime.datetime.now()
SVM_model.fit(X_train, y_train)
endtime = datetime.datetime.now()

print('Run time:', endtime - starttime)
print(list(zip(SVM_model.cv_results_['params'],SVM_model.cv_results_['mean_test_score'])))
print('Best params:', SVM_model.best_params_)
print('Best score: ', SVM_model.best_score_)


# The performance of the model on (train, validation) sets

SVM_model = SVM_model.best_estimator_

scoring = {
    'accuracy': 'accuracy',
    'precision_weighted': 'precision_weighted',
    'recall_weighted': 'recall_weighted',
    'f1_weighted': 'f1_weighted',
    'roc_auc_weighted': 'roc_auc_ovr_weighted',
    'aupr_weighted': make_scorer(average_precision_score, average='weighted', needs_proba=True)
}
cv = StratifiedKFold(n_splits = 5, shuffle = True)
scores = cross_validate(SVM_model, X_train, y_train, scoring=scoring, cv=cv, n_jobs=-1, return_train_score=False)

# The performance of the model on test sets

y_pred = SVM_model.predict(X_test)
y_pred_prob = SVM_model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision_weighted = precision_score(y_test, y_pred, average='weighted')
recall_weighted = recall_score(y_test, y_pred, average='weighted')
f1_weighted = f1_score(y_test, y_pred, average='weighted')
roc_auc_weighted = roc_auc_score(y_test, y_pred_prob, average='weighted', multi_class='ovr')
aupr_weighted = average_precision_score(y_test, y_pred_prob, average='weighted')


metrics = {
    "Metric": ["Accuracy", "Precision_weighted", "Recall_weighted", "F1-Score_weighted", "ROC AUC_weighted", "AUPR_weighted"],
    "Train_dataset(Mean)": [
        scores['test_accuracy'].mean(),
        scores['test_precision_weighted'].mean(),
        scores['test_recall_weighted'].mean(),
        scores['test_f1_weighted'].mean(),
        scores['test_roc_auc_weighted'].mean(),
        scores['test_aupr_weighted'].mean()
    ],
    "Train_dataset(Standard Deviation)": [
        scores['test_accuracy'].std(),
        scores['test_precision_weighted'].std(),
        scores['test_recall_weighted'].std(),
        scores['test_f1_weighted'].std(),
        scores['test_roc_auc_weighted'].std(),
        scores['test_aupr_weighted'].std()
    ],
    "Test Set": [
        accuracy,
        precision_weighted,
        recall_weighted,
        f1_weighted,
        roc_auc_weighted,
        aupr_weighted
    ]
}

df_SVM = pd.DataFrame(metrics)
print(df_SVM)
df_SVM.to_excel('../Out put/SVM_results.xlsx', index=False, engine='openpyxl')

# Save the model
f = open("../Trained model/SVM.pickle", 'wb')
pickle.dump(SVM_model,f, protocol = pickle.HIGHEST_PROTOCOL)
f.close()



# ## 2. Xgboost

# Simple test
y_train_ohe = OneHotEncoder().fit_transform(y_train.reshape(-1,1)).toarray()
y_test_ohe = OneHotEncoder().fit_transform(y_test.reshape(-1,1)).toarray()
XGB = XGBClassifier()
XGB.fit(X_train, y_train_ohe)
y_pred = XGB.predict(X_test)
print(accuracy_score(y_test_ohe, y_pred))

le = LabelEncoder()
y_train_le =  le.fit_transform(y_train)
y_test_le =  le.fit_transform(y_test)
le.classes_
XGB.fit(X_train, y_train_le)
y_pred = XGB.predict(X_test)
print(accuracy_score(y_test_le, y_pred))


# Parameter_tuning_1

param_grid = {
        "n_estimators": [50,100,200,300],
        "eta": [0.05, 0.1, 0,2, 0.3],
        "max_depth": [1,3,5,7,9],
        "colsample_bytree": [0.4,0.6,0.8,1],
        "min_child_weight": [1,2,4,6,8]
     }
XGB = XGBClassifier()
cv = StratifiedKFold(n_splits = 5, shuffle = True)
XGB_model = RandomizedSearchCV(XGB, param_distributions = param_grid, cv = cv, n_iter = 30, n_jobs = -1)
XGB_model.fit(X_train, y_train_le)

print(list(zip(XGB_model.cv_results_['params'],XGB_model.cv_results_['mean_test_score'])))
print(XGB_model.best_params_)
print(XGB_model.best_score_)


# Parameter_tuning_2
param_grid = {
        "n_estimators": [150, 200, 250],
        "eta": [0.05, 0.1, 0.2],
        "max_depth": [7, 8, 9],
        "colsample_bytree": [0.6, 0.8, 1],
        "min_child_weight": [4, 6, 8]
     }
XGB = XGBClassifier()
cv = StratifiedKFold(n_splits = 5, shuffle = True)
XGB_model = RandomizedSearchCV(XGB, param_distributions = param_grid, cv = cv, n_iter = 20, n_jobs = -1)
XGB_model.fit(X_train, y_train_le)

print(list(zip(XGB_model.cv_results_['params'],XGB_model.cv_results_['mean_test_score'])))
print(XGB_model.best_params_)
print(XGB_model.best_score_)


# The performance of the model on (train, validation) sets

XGB_model = XGB_model.best_estimator_


scoring = {
    'accuracy': 'accuracy',
    'precision_weighted': 'precision_weighted',
    'recall_weighted': 'recall_weighted',
    'f1_weighted': 'f1_weighted',
    'roc_auc_weighted': 'roc_auc_ovr_weighted',
    'aupr_weighted': make_scorer(average_precision_score, average='weighted', needs_proba=True)
}
cv = StratifiedKFold(n_splits = 5, shuffle = True)
scores = cross_validate(XGB_model, X_train, y_train_le, scoring=scoring, cv=cv, return_train_score=False)

# The performance of the model on test sets

y_pred = XGB_model.predict(X_test)
y_pred_prob = XGB_model.predict_proba(X_test)

accuracy = accuracy_score(y_test_le, y_pred)
precision_weighted = precision_score(y_test_le, y_pred, average='weighted')
recall_weighted = recall_score(y_test_le, y_pred, average='weighted')
f1_weighted = f1_score(y_test_le, y_pred, average='weighted')
roc_auc_weighted = roc_auc_score(y_test_le, y_pred_prob, average='weighted', multi_class='ovr')
aupr_weighted = average_precision_score(y_test_le, y_pred_prob, average='weighted')

# con_matrix = sm.confusion_matrix(y_test, y_pred, labels=['Coal', 'Construction', 'Road dust', 'Soil', 'Steel', 'Biomass'])
# print(con_matrix)
# report = sm.classification_report(y_test, y_pred)
# print(report)

metrics = {
    "Metric": ["Accuracy", "Precision_weighted", "Recall_weighted", "F1-Score_weighted", "ROC AUC_weighted", "AUPR_weighted"],
    "Train_dataset(Mean)": [
        scores['test_accuracy'].mean(),
        scores['test_precision_weighted'].mean(),
        scores['test_recall_weighted'].mean(),
        scores['test_f1_weighted'].mean(),
        scores['test_roc_auc_weighted'].mean(),
        scores['test_aupr_weighted'].mean()
    ],
    "Train_dataset(Standard Deviation)": [
        scores['test_accuracy'].std(),
        scores['test_precision_weighted'].std(),
        scores['test_recall_weighted'].std(),
        scores['test_f1_weighted'].std(),
        scores['test_roc_auc_weighted'].std(),
        scores['test_aupr_weighted'].std()
    ],
    "Test_dataset": [
        accuracy,
        precision_weighted,
        recall_weighted,
        f1_weighted,
        roc_auc_weighted,
        aupr_weighted
    ]
}

df_XGB = pd.DataFrame(metrics)
print(df_XGB)
df_XGB.to_excel('../Out put/XGB_results.xlsx', index=False, engine='openpyxl')

# save the model
f = open("../Trained model/Xgboost.pickle", 'wb')
pickle.dump(XGB_model,f, protocol = pickle.HIGHEST_PROTOCOL)
f.close()



# ## 3. RF

# Simple test
classifier = RandomForestClassifier(n_estimators=50, criterion='gini', random_state=42)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(accuracy_score(y_test, y_pred))


# Parameter_tuning
score_lt = []
for i in range(0,200,10):
    rfc = RandomForestClassifier(n_estimators=i+1)
    cv = StratifiedKFold(n_splits = 5, shuffle = True)
    score = cross_val_score(rfc, X_train, y_train, cv=cv).mean()
    score_lt.append(score)
score_max = max(score_lt)
print('Max score：{}'.format(score_max),
      'num：{}'.format(score_lt.index(score_max)*10+1))

# learning curve
x = np.arange(1,201,10)
plt.subplot(111)
plt.plot(x, score_lt, 'r-')
plt.show()

rfc = RandomForestClassifier(n_estimators=100)
param_grid = {'max_depth':np.arange(1,20)}
cv = StratifiedKFold(n_splits = 5, shuffle = True)
RF_model = GridSearchCV(rfc, param_grid, cv=cv, n_jobs= -1)
RF_model.fit(X_train, y_train)

print(list(zip(RF_model.cv_results_['params'],RF_model.cv_results_['mean_test_score'])))
best_param = RF_model.best_params_
best_score = RF_model.best_score_
print(best_param, best_score)

# The performance of the model on (train, validation) sets

RF_model = RF_model.best_estimator_

# scoring = ['accuracy','balanced_accuracy','precision_weighted', 'recall_weighted','f1_weighted', 'roc_auc_ovr_weighted', 'average_precision']
scoring = {
    'accuracy': 'accuracy',
    'precision_weighted': 'precision_weighted',
    'recall_weighted': 'recall_weighted',
    'f1_weighted': 'f1_weighted',
    'roc_auc_weighted': 'roc_auc_ovr_weighted',
    'aupr_weighted': make_scorer(average_precision_score, average='weighted', needs_proba=True)
}
cv = StratifiedKFold(n_splits = 5, shuffle = True)
scores = cross_validate(RF_model, X_train, y_train, scoring=scoring, cv=cv, return_train_score=False)

# The performance of the model on test sets

y_pred = RF_model.predict(X_test)
y_pred_prob = RF_model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision_weighted = precision_score(y_test, y_pred, average='weighted')
recall_weighted = recall_score(y_test, y_pred, average='weighted')
f1_weighted = f1_score(y_test, y_pred, average='weighted')
roc_auc_weighted = roc_auc_score(y_test, y_pred_prob, average='weighted', multi_class='ovr')
aupr_weighted = average_precision_score(y_test, y_pred_prob, average='weighted')

# con_matrix = sm.confusion_matrix(y_test, y_pred, labels=['Coal', 'Construction', 'Road dust', 'Soil', 'Steel', 'Biomass'])
# print(con_matrix)
# report = sm.classification_report(y_test, y_pred)
# print(report)

metrics = {
    "Metric": ["Accuracy", "Precision_weighted", "Recall_weighted", "F1-Score_weighted", "ROC AUC_weighted", "AUPR_weighted"],
    "Train_dataset(Mean)": [
        scores['test_accuracy'].mean(),
        scores['test_precision_weighted'].mean(),
        scores['test_recall_weighted'].mean(),
        scores['test_f1_weighted'].mean(),
        scores['test_roc_auc_weighted'].mean(),
        scores['test_aupr_weighted'].mean()
    ],
    "Train_dataset(Standard Deviation)": [
        scores['test_accuracy'].std(),
        scores['test_precision_weighted'].std(),
        scores['test_recall_weighted'].std(),
        scores['test_f1_weighted'].std(),
        scores['test_roc_auc_weighted'].std(),
        scores['test_aupr_weighted'].std()
    ],
    "Test Set": [
        accuracy,
        precision_weighted,
        recall_weighted,
        f1_weighted,
        roc_auc_weighted,
        aupr_weighted
    ]
}

df_RF = pd.DataFrame(metrics)
print(df_RF)
df_RF.to_excel('../Out put/RF_results.xlsx', index=False, engine='openpyxl')

# Save the model
f = open("../Trained model/RF_nodensity.pickle", 'wb')
pickle.dump(RF_model,f, protocol = pickle.HIGHEST_PROTOCOL)
f.close()




# ## 4. KNN

# Simple test
clf = neighbors.KNeighborsClassifier(n_neighbors = 15, weights = 'distance')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('Acurancy score:',accuracy_score(y_test, y_pred))



# Parameter_tuning_1
knn_est = Pipeline([('scaler', StandardScaler()), ('knn',neighbors.KNeighborsClassifier())])
params_grid = dict(knn__n_neighbors = [5, 10, 20, 30, 50, 80], knn__p = [1, 2], knn__weights = ['uniform', 'distance'])
cv = StratifiedKFold(n_splits = 5, shuffle = True)
KNN_model = RandomizedSearchCV(knn_est, param_distributions = params_grid, cv = cv, n_iter = 20, n_jobs = -1)
KNN_model.fit(X_train, y_train)

print(list(zip(KNN_model.cv_results_['params'],KNN_model.cv_results_['mean_test_score'])))
print(KNN_model.best_score_)
print(KNN_model.best_params_)


# Parameter_tuning_2
knn_est = Pipeline([('scaler', StandardScaler()), ('knn',neighbors.KNeighborsClassifier(weights = 'distance'))])
params_grid = dict(knn__n_neighbors = [1, 5, 10, 20], knn__p = [1, 2])
cv = StratifiedKFold(n_splits = 5, shuffle = True)
KNN_model = GridSearchCV(knn_est, param_grid = params_grid, cv = cv, n_jobs = -1)
KNN_model.fit(X_train, y_train)

print(list(zip(KNN_model.cv_results_['params'],KNN_model.cv_results_['mean_test_score'])))
print(KNN_model.best_score_)
print(KNN_model.best_params_)


# The performance of the model on (train, validation) sets

KNN_model = KNN_model.best_estimator_

scoring = {
    'accuracy': 'accuracy',
    'precision_weighted': 'precision_weighted',
    'recall_weighted': 'recall_weighted',
    'f1_weighted': 'f1_weighted',
    'roc_auc_weighted': 'roc_auc_ovr_weighted',
    'aupr_weighted': make_scorer(average_precision_score, average='weighted', needs_proba=True)
}
cv = StratifiedKFold(n_splits = 5, shuffle = True)
scores = cross_validate(KNN_model, X_train, y_train, scoring=scoring, cv=cv, return_train_score=False)



# The performance of the model on test sets

y_pred = KNN_model.predict(X_test)
y_pred_prob = KNN_model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision_weighted = precision_score(y_test, y_pred, average='weighted')
recall_weighted = recall_score(y_test, y_pred, average='weighted')
f1_weighted = f1_score(y_test, y_pred, average='weighted')
roc_auc_weighted = roc_auc_score(y_test, y_pred_prob, average='weighted', multi_class='ovr')
aupr_weighted = average_precision_score(y_test, y_pred_prob, average='weighted')

metrics = {
    "Metric": ["Accuracy", "Precision_weighted", "Recall_weighted", "F1-Score_weighted", "ROC AUC_weighted", "AUPR_weighted"],
    "Train_dataset(Mean)": [
        scores['test_accuracy'].mean(),
        scores['test_precision_weighted'].mean(),
        scores['test_recall_weighted'].mean(),
        scores['test_f1_weighted'].mean(),
        scores['test_roc_auc_weighted'].mean(),
        scores['test_aupr_weighted'].mean()
    ],
    "Train_dataset(Standard Deviation)": [
        scores['test_accuracy'].std(),
        scores['test_precision_weighted'].std(),
        scores['test_recall_weighted'].std(),
        scores['test_f1_weighted'].std(),
        scores['test_roc_auc_weighted'].std(),
        scores['test_aupr_weighted'].std()
    ],
    "Test Set": [
        accuracy,
        precision_weighted,
        recall_weighted,
        f1_weighted,
        roc_auc_weighted,
        aupr_weighted
    ]
}

df_KNN = pd.DataFrame(metrics)
print(df_KNN)
df_KNN.to_excel('../Out put/KNN_results.xlsx', index=False, engine='openpyxl')

# Save the model
with open("./Model/KNN.pickle", 'wb') as f:
    pickle.dump(KNN_model, f, protocol=pickle.HIGHEST_PROTOCOL)

