import os
import sys
import time
import gc
import random
import json
import pickle
import datetime
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, label_binarize
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import torch
from torch import nn
import torch.optim as optim
import torchvision
from torchvision import transforms, models, datasets




### SVM
f = open("../Trained model/SVM.pickle", 'rb')
model = pickle.load(f)
f.close 


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


y_pred = model.predict(X_test)

def array_to_dataframe(y_test, y_pred):
    mapping = {'Biomass':'Biomass burning', 'Coal':'Coal combustion', 'Construction':'Construction dust', 'Road dust':'Road dust', 'Soil':'Soil', 'Steel':'Steelmaking'}
    Dataframe = pd.DataFrame({'True_labels': y_test, 'Predicted_labels': y_pred})
    Dataframe['True_labels'] = Dataframe['True_labels'].replace(mapping)
    Dataframe['Predicted_labels'] = Dataframe['Predicted_labels'].replace(mapping)
    return Dataframe

df = array_to_dataframe(y_test, y_pred)

labels = ['Soil', 'Road dust', 'Construction', 'Coal', 'Biomass', 'Steel']
disp_labels = ['Soil', 'Road dust', 'Construction dust', 'Coal combustin', 'Biomass burning', 'Steelmaking']

con_matrix = confusion_matrix(y_test, y_pred, labels = labels)


def con_matrix_normied_plot(data, colorbar_ticks = [0.2, 0.4, 0.6, 0.8], fig_path = False):
    
    labels = ['Soil', 'Road dust', 'Construction dust', 'Coal combustion', 'Biomass burning', 'Steelmaking']
    con_matrix = confusion_matrix(data['True_labels'], data['Predicted_labels'], labels = labels)
    con_matrix_normalized = confusion_matrix(data['True_labels'], data['Predicted_labels'], labels = labels, normalize='true')
                      
    plt.figure(figsize=(10, 10))
    plt.xticks(np.arange(len(labels)), labels=labels, rotation=30, rotation_mode="anchor", ha="right", fontsize = 25)
    plt.xlabel('Predicted source', fontsize = 30)
    plt.yticks(np.arange(len(labels)), labels=labels, fontsize = 25)
    plt.ylabel('True source', fontsize = 30)
    # plt.title("Harvest of local farmers (in tons/year)")
    f1 = lambda x: '%.2f%%' % (x * 100)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i == j:
                text = plt.text(j, i, f1(con_matrix_normalized[i, j]) + "\n" + "(" + str(con_matrix[i, j]) + ")", ha="center", va="center", color="b", fontsize = 22)
            else:
                text = plt.text(j, i, f1(con_matrix_normalized[i, j]) + "\n" + "(" + str(con_matrix[i, j]) + ")", ha="center", va="center", color="w", fontsize = 22)


    
    plt.imshow(con_matrix_normalized)
    
    cax = plt.axes([0.95, 0.12, 0.04, 0.76])
    colorbar = plt.colorbar(cax= cax)
    colorbar.ax.set_yticks(colorbar_ticks)
    colorbar.ax.set_yticklabels(['{:.0%}'.format(x) for x in colorbar_ticks], fontsize = 22)
    #plt.tight_layout()
    if fig_path:
        plt.savefig(fig_path, dpi=500, bbox_inches='tight') # 通过 bbox_inches 参数确保保存的图像包含所有的绘图元素
    plt.show()


con_matrix_normied_plot(df, fig_path='../Out put/SVM-confusion_matrix.png')




def multi_class_ROC(y_test, predict_proba, model_classes,fig_path = False):
    y_test_bin = label_binarize(y_test, classes=model_classes)
    y_test_pred_proba = predict_proba
    # model_classes = list(model_classes)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    j = 0
    for i in model_classes:
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, j], y_test_pred_proba[:, j])
        roc_auc[i] = auc(fpr[i], tpr[i])
        j += 1

    # Average result
    fpr["average"], tpr["average"], _ = roc_curve(y_test_bin.ravel(), y_test_pred_proba.ravel())
    roc_auc["average"] = auc(fpr["average"], tpr["average"])

    # Plot
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr["average"], tpr["average"],
             label='Average ROC curve (AUC = {0:0.2f})'
                   ''.format(roc_auc["average"]),
             color='red', linestyle=':', linewidth=4)

    colors = ['aqua', 'darkorange', 'cornflowerblue', 'cyan', 'greenyellow', 'dodgerblue']
    for i, color in zip(model_classes, colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='{0} (AUC = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize = 30)
    plt.ylabel('True Positive Rate', fontsize = 30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    # plt.title('multi-calss ROC')
    plt.legend(loc="lower right", fontsize = 18)
    if fig_path:
        plt.savefig(fig_path, dpi=500, bbox_inches='tight')
    plt.show()

sources = ['Biomass burning', 'Coal combustion', 'Construction dust', 'Road dust', 'Soil', 'Steelmaking']
y_pred_proba = model.predict_proba(X_test)
multi_class_ROC(df['True_labels'],y_pred_proba,sources, fig_path = '../Out put/SVM-ROC.png')




### XGBoost

f = open("../Trained model/Xgboost.pickle", 'rb')
model = pickle.load(f)
f.close()

le = LabelEncoder()
y_test_le =  le.fit_transform(y_test)

y_pred = model.predict(X_test)
df = array_to_dataframe(y_test, le.inverse_transform(y_pred))
con_matrix_normied_plot(df, fig_path='../Out put/XGBoost-confusion_matrix.png')

y_pred_proba = model.predict_proba(X_test)
sources = ['Biomass burning', 'Coal combustion', 'Construction dust', 'Road dust', 'Soil', 'Steelmaking']
multi_class_ROC(df['True_labels'],y_pred_proba,sources, fig_path = '../Out put/XGBoost-ROC.png')



### ResNet

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

df2 = array_to_dataframe(le.inverse_transform(y_true), le.inverse_transform(y_pred))
print(df2)

con_matrix_normied_plot(df2, fig_path='../Out put/ResNet-confusion_matrix.png')

sources = ['Biomass burning', 'Coal combustion', 'Construction dust', 'Road dust', 'Soil', 'Steelmaking']
multi_class_ROC(df2['True_labels'],df.values,sources, fig_path = '../Out put/ResNet-ROC.png')




### KNN

f = open("../Trained model/KNN.pickle", 'rb')
model = pickle.load(f)
f.close 

y_pred = model.predict(X_test)
df = array_to_dataframe(y_test, y_pred)
con_matrix_normied_plot(df, fig_path='../Out put/KNN-confusion.png')


y_pred_proba = model.predict_proba(X_test)
sources = ['Biomass burning', 'Coal combustion', 'Construction dust', 'Road dust', 'Soil', 'Steelmaking']
multi_class_ROC(df['True_labels'],y_pred_proba,sources, fig_path = '../Out put/KNN-ROC.png')


### RF

f = open("../Trained model/RF.pickle", 'rb')
model = pickle.load(f)
f.close

y_pred = model.predict(X_test)
df = array_to_dataframe(y_test, y_pred)
con_matrix_normied_plot(df, fig_path='../Out put/RF-confusion_matrix.png')

y_pred_proba = model.predict_proba(X_test)
sources = ['Biomass burning', 'Coal combustion', 'Construction dust', 'Road dust', 'Soil', 'Steelmaking']
multi_class_ROC(df['True_labels'],y_pred_proba,sources, fig_path = '../Out put/RF-ROC.png')

