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


class model_creator(nn.Module):
    def __init__(self,model_type, pretrained, device):
        super(model_creator, self).__init__()
        self.model_type = model_type
        self.pretrained_or_not = pretrained
        self.device = device
        
        if(model_type == "18"):
            self.model = models.resnet18(pretrained = pretrained).to(device)
        elif(model_type == "50"):
            self.model = models.resnet50(pretrained = pretrained).to(device)
        
        self.model.fc = nn.Linear(self.model.fc.in_features,5);

    def forward(self, x):
        return self.model(x)
    
    def evaluate(self, test_loader):
        correct = 0;
        total_samples = 0
        self.model.eval();
        with torch.no_grad():
            for input, labels in tqdm(test_loader):
                input = input.to(self.device)
                labels = labels.to(self.device, dtype = torch.long)
                forward_output = self.model(input);
                prediction = torch.argmax(forward_output, dim=1)
                # print(prediction)
                correct += torch.sum(prediction == labels).item()
                total_samples += prediction.shape[0]
                # print(torch.sum(prediction == labels).item())
        return correct / total_samples


    def train_and_eval(self, train_loader, test_loader, epoch):
        optim = torch.optim.Adam(params = self.model.parameters(), lr = 3e-3)
        loss_fcn = nn.CrossEntropyLoss()
        train_accs = list();
        test_accs = list();
        epochs = list();
        correct = 0;
        best_acc = 0;
        total_samples = 0
        for i in range(1 , epoch + 1):
            print({"Now epoch: ": i})
            self.model.train();
            for input,labels in tqdm(train_loader):
                input = input.to(self.device);
                labels = labels.to(self.device, dtype=torch.long)
                optim.zero_grad();
                # self.model.zero_grad()
                forward_output = self.model(input);
                prediction = torch.argmax(forward_output, dim = 1)
                loss = loss_fcn(forward_output, labels)
                loss.backward();
                optim.step()
                correct += torch.sum(prediction == labels).item()
                total_samples += prediction.shape[0]
            train_acc = correct / total_samples
            test_acc = self.evaluate(test_loader)
            epochs.append(i)
            train_accs.append(train_acc * 100.0)
            test_accs.append(test_acc * 100.0)
            if(test_acc > best_acc):
                best_acc = test_acc
                best_epoch = i
                if test_acc > 0.82:
                    fileName = "../model_depository/" + "_" + \
                        self.model_type + "_" + \
                        str(test_acc * 100.0) + ".pt"
                    # state_dict = copy.deepcopy(self.state_dict())
                    torch.save(self, fileName)
                    # self.evaluate(device,testing)
            print({"now epoch: ": i, "loss: ": loss.item(), "training accuracy: ": train_acc, "testing accuracy: ": test_acc})

        return epochs, train_accs, test_accs


def plot(epoch, train, test, pretrained):
    if pretrained == True:
        plt.suptitle("Accruray curve of pretrained model")
    else:
        plt.suptitle("Accruray curve of non-pretrained model")
    plt.xlabel("epoch")
    plt.ylabel("Accuracy(%)")
    plt.plot(epoch, train, "r")
    plt.plot(epoch, test, "g")




if __name__ == "__main__":


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    np.random.seed(123)
    random.seed(123)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    train_data, train_label = dataloader.getData("train")
    test_data,test_label = dataloader.getData("test")
    # print(dataloader.getData("train").shape)
    print(train_data.shape)
    print(train_label.shape)

    train_dataset = RetinopathyLoader("./data/", "train");
    test_dataset = RetinopathyLoader("./data/", "test");

    train_loader = DataLoader(train_dataset,batch_size = 30, shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size = 5, shuffle=False)
    print(train_dataset.__getitem__(0)[0].shape)

    model = model_creator("50",True,device);
    model.to(device)
    epoch, train_acc, test_acc = model.train_and_eval(train_loader, test_loader, 10);
    # test = model.evaluate(test_loader)
    plot(epoch,train_acc,test_acc,True);
    plt.legend(["Train", "Test"])
    plt.show()    




