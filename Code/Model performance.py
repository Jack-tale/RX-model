import os
import gc
import datetime
import random
import sys
import copy
import json
import warnings
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.optim as optim
import torchvision
from torchvision import transforms, models, datasets
from sklearn.metrics import accuracy_score, f1_score, recall_score, average_precision_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.model_selection import StratifiedShuffleSplit, RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler, label_binarize, LabelEncoder
import xgboost as xgb
from xgboost.sklearn import XGBClassifier



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




### SVM


f = open("../Trained model/SVM.pickle", 'rb')
model = pickle.load(f)
f.close()


y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, digits=3)
print(report)



y_prob = model.predict_proba(X_test)
y_test_bin = label_binarize(y_test, classes = ['Biomass', 'Coal', 'Construction', 'Road dust', 'Soil', 'Steel'])

# caculate AUPR
aupr_per_class = average_precision_score(y_test_bin, y_prob, average=None)
print(f"AUPR per class: {aupr_per_class}")





# ### XGBoost

f = open("../Trained model/Xgboost.pickle", 'rb')
model_2 = pickle.load(f)
f.close()


le = LabelEncoder()
y_test_le =  le.fit_transform(y_test)


y_pred = model_2.predict(X_test)

report = classification_report(y_test_le, y_pred, digits=3)
print(report)


y_pred_proba = model_2.predict_proba(X_test)
y_test_bin = label_binarize(y_test_le, classes = [0, 1, 2, 3, 4, 5])
aupr_per_class = average_precision_score(y_test_bin, y_pred_proba, average=None)
print(f"AUPR per class: {aupr_per_class}")



# ### RF

f = open("../Trained model/RF.pickle", 'rb')
model_3 = pickle.load(f)
f.close


y_pred = model_3.predict(X_test)
y_pred_proba = model_3.predict_proba(X_test)
report = classification_report(y_test, y_pred, digits=3)
print(report)


y_test_bin = label_binarize(y_test, classes = ['Biomass', 'Coal', 'Construction', 'Road dust', 'Soil', 'Steel'])
aupr_per_class = average_precision_score(y_test_bin, y_pred_proba, average=None)
print(f"AUPR per class: {aupr_per_class}")




# ### KNN

f = open("../Trained model/KNN.pickle", 'rb')
model_4 = pickle.load(f)
f.close 

y_pred = model_4.predict(X_test)
y_pred_proba = model_4.predict_proba(X_test)
report = classification_report(y_test, y_pred, digits=3)
print(report)


y_test_bin = label_binarize(y_test, classes = ['Biomass', 'Coal', 'Construction', 'Road dust', 'Soil', 'Steel'])
aupr_per_class = average_precision_score(y_test_bin, y_pred_proba, average=None)
print(f"AUPR per class: {aupr_per_class}")



# ### ResNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load("../Trained model/ResNet.pth")
model.to(device)
model.eval()

data_dir = '../Row data/Training data/'
test_dir = data_dir + 'test/'

val_tf = transforms.Compose([
         transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

image_datasets = datasets.ImageFolder(test_dir, val_tf)
dataloader = torch.utils.data.DataLoader(image_datasets, batch_size = 64, num_workers = 11)

running_loss = 0.0
running_corrects = 0

y_true = []
y_pred = []
df = pd.DataFrame()

for inputs, labels in dataloader:
    inputs = inputs.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(inputs)

    _, preds = torch.max(outputs, 1)
    probs = nn.functional.softmax(outputs, dim=1)
    class_probabilities = probs.detach().cpu().numpy()
    

    y_true.extend(labels.detach().cpu().numpy())
    y_pred.extend(preds.detach().cpu().numpy())
    df = df._append(pd.DataFrame(class_probabilities), ignore_index=True)




report = classification_report(y_true, y_pred, digits=3)
print(report)


y_test_bin = label_binarize(y_test_le, classes = [0, 1, 2, 3, 4, 5])
aupr_per_class = average_precision_score(y_test_bin, df.values, average=None)
print(f"AUPR per class: {aupr_per_class}")





# ### RX
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load("../Trained model/ResNet.pth")
model.to(device)
model.eval()

data_dir = '../Row data/Training data/'
train_dir = data_dir + 'train/'
test_dir = data_dir + 'test/'

val_tf = transforms.Compose([
         transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

image_datasets = datasets.ImageFolder(train_dir, val_tf)
dataloader = torch.utils.data.DataLoader(image_datasets, batch_size = 64, num_workers = 11)

df = pd.DataFrame()
for inputs, labels in dataloader:
    inputs = inputs.to(device)
    labels = labels.to(device)
    with torch.no_grad():
        outputs = model(inputs)
    probs = nn.functional.softmax(outputs, dim=1)
    class_probabilities = probs.detach().cpu().numpy()
    df = df._append(pd.DataFrame(class_probabilities), ignore_index=True)
image_datasets = datasets.ImageFolder(test_dir, val_tf)
dataloader = torch.utils.data.DataLoader(image_datasets, batch_size = 64, num_workers = 11)

df2 = pd.DataFrame()
for inputs, labels in dataloader:
    inputs = inputs.to(device)
    labels = labels.to(device)
    with torch.no_grad():
        outputs = model(inputs)
    probs = nn.functional.softmax(outputs, dim=1)
    class_probabilities = probs.detach().cpu().numpy()
    df2 = df2._append(pd.DataFrame(class_probabilities), ignore_index=True)

# Merging with EDS and size data
pickle_file = '../Row data/train_test_data_2.pickle'
with open(pickle_file, 'rb') as f:
    pickle_data = pickle.load(f)
    X_train = pickle_data['train_dataset']
    y_train = pickle_data['train_labels']
    X_test = pickle_data['test_dataset']
    y_test = pickle_data['test_labels']
    del pickle_data
    gc.collect()
print('The data has been loaded')

X_train = np.hstack([np.array(df),X_train])
X_test = np.hstack([np.array(df2),X_test])


le = LabelEncoder()
y_train_le =  le.fit_transform(y_train)
y_test_le =  le.fit_transform(y_test)
le.classes_

f = open("../Trained model/RX.pickle", 'rb')
model_6 = pickle.load(f)
f.close()

y_pred = model_6.predict(X_test)
report = classification_report(y_test_le, y_pred, digits=3)
print(report)

y_pred_proba = model_6.predict_proba(X_test)
y_test_bin = label_binarize(y_test_le, classes = [0, 1, 2, 3, 4, 5])
aupr_per_class = average_precision_score(y_test_bin, y_pred_proba, average=None)
print(f"AUPR per class: {aupr_per_class}")