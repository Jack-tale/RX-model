import pandas as pd
import numpy as np
import pickle
import os
import glob
import imageio
import time
import random
import sys
import copy
import json
import shutil
from PIL import Image
import gc
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.optim as optim
import torchvision
from torchvision import transforms, models, datasets
import warnings


# ## ResNet model 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load("../Trained model/ResNet.pth")
# The "ResNet.pth" file is too large to upload to GitHub, so it has been uploaded to the Figshare platform (https://doi.org/10.6084/m9.figshare.28137446.v1)

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
