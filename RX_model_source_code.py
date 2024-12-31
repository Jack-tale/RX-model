#!/usr/bin/env python
# coding: utf-8

# ## Processing of EDS and Particle Size Data

import pandas as pd
import numpy as np
import pickle
import os
import glob
from sklearn.model_selection import train_test_split
import imageio
import time
import random
import sys
import copy
import json
from PIL import Image
import shutil


data = pd.read_excel("./Training data/6_source_data.xlsx", sheet_name = 0) #Reading EDS and size data containing six pollution sources
data.insert(0, 'id_code', list(range(len(data))))
dataset = np.array(data)
labels = np.array(data.iloc[:,119])
train_dataset, test_dataset, train_labels, test_labels = train_test_split(dataset, labels, test_size = .2, stratify = labels)

# Sorting training and test set data by id_code
train_data = pd.DataFrame(train_dataset)
train_data.columns = data.columns
train_data = train_data.sort_values(by='id_code')
test_data = pd.DataFrame(test_dataset)
test_data.columns = data.columns
test_data = test_data.sort_values(by='id_code')

writer = pd.ExcelWriter('./Training data/train_test_row_data.xlsx')
train_data.to_excel(writer, sheet_name = 'train_data', index = False)
test_data.to_excel(writer, sheet_name = 'test_data', index = False)
writer._save()

# Selecting 5 size parameters and 23 elements
train_data_2 = np.array(train_data.iloc[:,list([12, 13, 14, 15, 16])+list(range(96,119))])
test_data_2 = np.array(test_data.iloc[:,list([12, 13, 14, 15, 16])+list(range(96,119))])
train_labels_2 = np.array(train_data.iloc[:,119])
test_labels_2 = np.array(test_data.iloc[:,119])

# Data Saving
pickle_file = './Training data/train_test_data_2.pickle'
if not os.path.isfile(pickle_file):
    print('Saving data to pickle file ...')
    try:
        with open(pickle_file, 'wb') as pfile:
            pickle.dump(
                {
                    'train_dataset':train_data_2,
                    'train_labels': train_labels_2,
                    'test_dataset': test_data_2,
                    'test_labels': test_labels_2
                },
                pfile, pickle.HIGHEST_PROTOCOL
            )
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise
print('Data cached in pickle file.')


# ## Organizing Microscopic Images


path = './source_data'
files = os.listdir(path) #['Biomass', 'Coal', 'Construction', 'Road dust', 'Soil', 'Steel']

# Consolidating images into one folder and renaming them to correspond with predefined particle names in the Excel file
for file_folder in files:
    Summarize_pic_folder = './Training data'
    if not os.path.exists(os.path.join(Summarize_pic_folder, 'Summarize_pic_folder')):
        os.makedirs(os.path.join(Summarize_pic_folder, 'Summarize_pic_folder'))
    file_index = "01"
    file_subfolder = os.listdir(os.path.join(path, file_folder, file_index))
    none_xlsx_file_subfolder = [file for file in file_subfolder if not file.endswith('.xlsx')]
    for pic_folder_name in none_xlsx_file_subfolder:
        pic_folder = os.path.join(path, file_folder, file_index, pic_folder_name)
        for image in os.listdir(pic_folder):
            source_path = os.path.join(pic_folder, image)
            destination_name = os.path.join(os.path.join(Summarize_pic_folder, 'Summarize_pic_folder'), f"{file_folder}_{pic_folder_name}_{image[:-4].zfill(5)}"+".png")
            shutil.copyfile(source_path, destination_name)

folder_list = [str(i) for i in range(len(files))]
source_dict = {k: v for k, v in zip(folder_list, files)}

train_data = pd.read_excel('./Training data/train_test_row_data.xlsx', sheet_name = "train_data")
test_data = pd.read_excel('./Training data/train_test_row_data.xlsx', sheet_name = "test_data")

train_data, test_data = train_data[['Part #', 'Source', 'file_name']], test_data[['Part #', 'Source', 'file_name']]

for key, source_label in source_dict.items():
    train_folder = './Training data/train'
    test_folder = './Training data/test'
    if not os.path.exists(os.path.join(train_folder, key)):
        os.makedirs(os.path.join(train_folder, key))
    if not os.path.exists(os.path.join(test_folder, key)):
        os.makedirs(os.path.join(test_folder, key))
    # train pic
    for index, row in train_data[train_data['Source'] == source_label].iterrows():
        img_name = source_label + '_' + row['file_name'] + '_' + str(row['Part #']).zfill(5)
        src_path = os.path.join(Summarize_pic_folder, 'Summarize_pic_folder', img_name + '.png')
        dst_path = os.path.join(train_folder, key, img_name + '.png')
        shutil.copyfile(src_path, dst_path)  
    # test pic
    for index, row in test_data[test_data['Source'] == source_label].iterrows():
        img_name = source_label + '_' + row['file_name'] + '_' + str(row['Part #']).zfill(5)
        src_path = os.path.join(Summarize_pic_folder, 'Summarize_pic_folder', img_name + '.png')
        dst_path = os.path.join(test_folder, key, img_name + '.png')
        shutil.copyfile(src_path, dst_path)
shutil.rmtree('./Training data/Summarize_pic_folder')


# ## ResNet model training


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import torch
from torch import nn
import torch.optim as optim
import torchvision
from torchvision import transforms, models, datasets
import warnings
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



data_dir = './Training data/'
train_dir = data_dir + 'train/'
test_dir = data_dir + 'test/'

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomRotation(90),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.RandomVerticalFlip(p = 0.5),
        transforms.ColorJitter(brightness = 0.2, contrast = 0.1, saturation = 0.1, hue = 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'test']}



# Viewing Images

batch_size = 16
num_workers = 12

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size = batch_size, shuffle = True, num_workers=num_workers) for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes
files = ['Biomass', 'Coal', 'Construction', 'Road dust', 'Soil', 'Steel']
folder_list = [str(i) for i in range(6)]
source_dict = {k: v for k, v in zip(folder_list, files)}

def im_convert(tensor):
    """展示数据"""
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze() 
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    return image

fig = plt.figure(figsize=(20, 12))
columns = 4
rows = 2
dataiter = iter(dataloaders['test'])
inputs, classes = next(dataiter)

for idx in range (min(columns*rows, len(inputs))):
    ax = fig.add_subplot(rows, columns, idx+1, xticks=[], yticks=[])
    plt.imshow(im_convert(inputs[idx]))
    ax.set_title(source_dict[str(int(class_names[classes[idx]]))])
plt.show()



# Model training

def initialize_model(num_classes, use_pretrained = True):
    model_ft = models.resnet152(pretrained = use_pretrained)
    for param in model_ft.parameters():
        param.requires_grad = False
    for param in model_ft.layer4.parameters():
        param.requires_grad = True
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes),
                                nn.LogSoftmax(dim = 1))
    
    return model_ft

model_ft = initialize_model(6, use_pretrained=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_ft = model_ft.to(device)

print("Params to learn:")
params_to_update = []
for name,param in model_ft.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)
        print("\t", name)


# Optimizer Configuration
optimizer_ft = optim.Adam(params_to_update, lr = 1e-2)
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size = 7, gamma = 0.1)
criterion = nn.NLLLoss()

def train_model(model, dataloaders, criterion, optimizer, num_epochs = 25, filename = 'checkpoint.pth'):
    since = time.time()
    best_acc = 0
    """
    checkpoint = torch.load(filename)
    best_acc = checkpoint['best_acc']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.class_to_idx = checkpoint['mapping']
    """
    model.to(device)
    
    val_acc_history = []
    train_acc_history = []
    train_losses = []
    test_losses = []
    LRs = [optimizer.param_groups[0]['lr']] 
    
    best_model_wts = copy.deepcopy(model.state_dict())
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-'*10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            
            # Iterating through the data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Clearing gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1) #Finding the maximum value and its corresponding index in the predicted result
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            time_elapsed = time.time() - since
            print('Time elapsed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('{} Loss: {:.4f}  Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                state = {
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer':optimizer.state_dict()
                }
                torch.save(state, filename)
            if phase == 'test':
                val_acc_history.append(epoch_acc)
                test_losses.append(epoch_loss)
                scheduler.step(epoch_loss)
            if phase == 'test':
                train_acc_history.append(epoch_acc)
                train_losses.append(epoch_loss)
        
        print('Optimizer learning rate : {:.7f}'.format(optimizer.param_groups[0]['lr']))
        LRs.append(optimizer.param_groups[0]['lr'])
        print()
        
    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history, test_losses, train_losses, LRs

model_ft, val_acc_history, train_acc_history, test_losses, train_losses, LRs = train_model(model_ft, dataloaders, criterion, optimizer_ft, num_epochs = 50)
torch.save(model_ft, "./Model/6_Sources_identify_model_0310.pth")


# ## ResNet-XGBoost Training

import pickle
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import gc
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.model_selection import StratifiedShuffleSplit, RandomizedSearchCV, GridSearchCV,cross_val_score
import datetime
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, label_binarize


## Calculating the probability values of single particle sources using the trained ResNet model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load("./Model/6_Sources_identify_model_0310.pth")
model.to(device)
model.eval()

data_dir = './Training data/'
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
pickle_file = './Training data/train_test_data_2.pickle'
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



# XGBoost model training
le = LabelEncoder()
y_train_le =  le.fit_transform(y_train)
y_test_le =  le.fit_transform(y_test)
le.classes_

XGB = XGBClassifier()
XGB.fit(X_train, y_train_le)
y_pred = XGB.predict(X_test)
accuracy_score(y_test_le, y_pred)


starttime = datetime.datetime.now()
param_grid = {
        "n_estimators": [100, 200, 300, 500],
        "eta": [0.05, 0.1, 0,2, 0.3],
        "max_depth": [3,4,5,6,7,8],
        "colsample_bytree": [0.4,0.6,0.8,1],
        "min_child_weight": [1,2,4,6,8]
     }
XGB = XGBClassifier()

GS = RandomizedSearchCV(XGB, param_distributions = param_grid, cv = 5, n_iter = 50, n_jobs = -1)
GS.fit(X_train, y_train_le)
endtime = datetime.datetime.now()
print('Run time:', endtime - starttime)
print("Best params：:", GS.best_params_)
print("Best score:", GS.best_score_)

y_pred = GS.predict(X_test)
accuracy_score(y_test_le, y_pred)



starttime = datetime.datetime.now()
param_grid = {
        "n_estimators": [200, 300, 350, 400],
        "eta": [0.03, 0.05, 0.1, 0,2],
        "max_depth": [6,7,8,9, 10],
        "colsample_bytree": [0.3, 0.4,0.6,0.8,1],
        "min_child_weight": [3,4,5,7]
     }
XGB = XGBClassifier()

GS = RandomizedSearchCV(XGB, param_distributions = param_grid, cv = 4, n_iter = 50, n_jobs = -1)
GS.fit(X_train, y_train_le)
endtime = datetime.datetime.now()
print('Run time:', endtime - starttime)
print("Best params：:", GS.best_params_)
print("Best score:", GS.best_score_)


f = open("./Model/ResNet-Xgboost_model.pickle", 'wb')
pickle.dump(GS.best_estimator_,f, protocol = pickle.HIGHEST_PROTOCOL)
f.close()
