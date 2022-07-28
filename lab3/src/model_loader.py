from ast import arg
from turtle import forward
import pandas as pd
import numpy as np
import torch
import secrets
import random
import dataloader
from dataloader import RetinopathyLoader
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torchvision import transforms, models
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix
import time
from sklearn.metrics import precision_recall_fscore_support

# +
from prefetch_generator import BackgroundGenerator

class DataLoader_pro(DataLoader):    
    def __iter__(self):
        return BackgroundGenerator(super().__iter__());


# +
def plot_confusion_matrix(model,testing, device, model_type, pretrained):
    plt.figure(figsize=(10, 8))
    print("Plotting confusion matrix...")
    preds = list();
    labels = list();
    with torch.no_grad():
        for input,label in tqdm(testing):
            input = input.to(device);
            label = label.to(device, dtype=torch.long)
            pred = model(input)
            pred = torch.argmax(pred, dim=1)
            labels.extend(label.tolist())
            preds.extend(pred.tolist())
    
    ## Calculate confusion matrix
    cmatrix = confusion_matrix(labels,preds)
    p,r,f,s = precision_recall_fscore_support(labels, preds, average='macro')
    ## Normalize
    cmatrix = cmatrix / cmatrix.sum(axis=1)[np.newaxis].T

    ## Plot
    textcolors = ("black", "white")
    fig, ax = plt.subplots()
    img = ax.imshow(cmatrix, cmap=plt.cm.Blues)
    for i in range(cmatrix.shape[0]):
        for j in range(cmatrix.shape[1]):
            ax.text(
                j, i, "{:.2f}".format(cmatrix[i, j]), 
                ha="center", va="center", 
                color=textcolors[cmatrix[i, j] > 0.5]
            )

    plt.colorbar(img)
#     plt.title("Normalized Confusion Matrix (ResNet{})".format(args.model[-2:]))
    plt.title("Normalized Confusion Matrix (" + model_type + ")");
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    pretrained_str = "w" if pretrained else "wo"

#     plt.savefig("confusion_matrix.png")
    plt.savefig("cm_{}_{}_pretrained_{}_{}.png".format(model_type, pretrained_str, p * 100, r * 100))


# +
def evaluate(model,test_loader,device):
    correct = 0;
    total_samples = 0
    model.eval();
    with torch.no_grad():
        for input, labels in tqdm(test_loader):
            input = input.to(device)
            labels = labels.to(device)
            forward_output = model(input);
            prediction = torch.argmax(forward_output, dim=1)
            correct += torch.sum(prediction == labels).item()
            total_samples += prediction.shape[0]
    return correct / total_samples

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    test_dataset = RetinopathyLoader(os.getcwd() + "/data/", "test");
    test_loader = DataLoader_pro(test_dataset,batch_size = 20, shuffle=False)
    # model = models.resnet50(pretrained = False);
    # model.load(torch.load(os.getcwd() + "/model_depository/ResNet50_weight_82.12099644128114.pt"))
    model = torch.load(os.getcwd() + "/model_depository/ResNet50_82.23487544483986.pt")
#     model = nn.DataParallel(model)
    model = model.to(device);
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # plot_confusion_matrix(model,test_loader,device,"ResNet50",True)
    print(evaluate(model,test_loader,device))
# -




