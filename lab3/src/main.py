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
# from my_dataloader import getData


# print(os.getcwd())
# _,_  = getData("train")
# -

def model_evaluation(model,test,device):
    for batch_idx, (data, target) in enumerate(test):
        data = data.to(device)
        target = target.to(device)
        model.eval();
        prediction = model(data)
        # print({"prediction type: ":prediction.type()});
        preds = torch.argmax(prediction, dim=1)
        # print({"preds type: ":preds.type()});

        accuracy = torch.sum(preds == target).item() / len(target)
        print({"Testing accuracy: ": accuracy})


class model_creator(nn.Module):
    def __init__(self,model_type, pretrained, device):
        super(model_creator, self).__init__()
        self.model_type = model_type
        self.pretrained_or_not = pretrained
        self.device = device
        
        if(model_type == "18"):
            self.model_type = "ResNet18"
            self.model = models.resnet18(pretrained = pretrained).to(device)
        elif(model_type == "50"):
            self.model_type = "ResNet50"
            self.model = models.resnet50(pretrained = pretrained).to(device)
        if pretrained:
            for param in self.model.parameters():
                param.requires_grad = False 
        self.model.fc = nn.Linear(self.model.fc.in_features,5);
        self.model = nn.DataParallel(self.model)

    def forward(self, x):
        return self.model(x)
    
    def evaluate(self, test_loader):
        correct = 0;
        total_samples = 0
        self.model.eval();
        all_labels = list()
        all_preds = list()

        with torch.no_grad():
#             for batch_idx, (input, labels) in enumerate(test_loader):
            for input, labels in tqdm(test_loader):
                input = input.to(self.device)
                labels = labels.to(self.device, dtype = torch.long)
                forward_output = self.model(input);
                preds = torch.argmax(forward_output, dim=1)
                correct += torch.sum(preds == labels).item()
                total_samples += preds.shape[0]
                all_labels.extend(labels.tolist())
                all_preds.extend(preds.tolist())
        return correct / total_samples, all_preds, all_labels


    def train_and_eval(self, train_loader, test_loader, epoch, training_type, lr, weight_decay):
        if training_type == "linear-probe":
            params_to_update = [];
            for name, param in self.model.named_parameters():
                if param.requires_grad :
                    params_to_update.append(param);
#                     optim = torch.optim.SGD(params_to_update,lr = lr ,momentum = 0.9, weight_decay = 5e-4 )
                    optim = torch.optim.SGD(params_to_update,lr = lr ,momentum = 0.9, weight_decay = weight_decay)
        elif training_type == "fine-tune":
            for param in self.model.parameters():
                param.requires_grad = True;
#                 optim = torch.optim.SGD(self.model.parameters(),lr = lr ,momentum = 0.9, weight_decay = 5e-4 )
                optim = torch.optim.SGD(self.model.parameters(),lr = lr ,momentum = 0.9, weight_decay = weight_decay)
        optim = nn.DataParallel(optim)
        loss_fcn = nn.CrossEntropyLoss()
        train_accs = list();
        test_accs = list();
        epochs = list();
        correct = 0;
        best_acc = 0;
        total_samples = 0
        
        for i in range(1 , epoch + 1):
            self.model.train();
#             for batch_idx, (input, labels) in enumerate(train_loader):
            for input,labels in tqdm(train_loader):
                input = input.to(self.device, non_blocking = True);
                labels = labels.to(self.device, dtype=torch.long, non_blocking = True)
                optim.module.zero_grad();
                forward_output = self.model(input);
                prediction = torch.argmax(forward_output, dim = 1)
                loss = loss_fcn(forward_output, labels)
                loss.backward();
                optim.module.step()
                correct += torch.sum(prediction == labels).item()
                total_samples += prediction.shape[0]
            train_acc = correct / total_samples
            test_acc, test_preds,test_labels = self.evaluate(test_loader)
            epochs.append(i)
            train_accs.append(train_acc * 100.0)
            test_accs.append(test_acc * 100.0)
            if(test_acc > best_acc):
                best_acc = test_acc
                best_epoch = i
                best_labels = test_labels
                best_preds = test_preds
                if test_acc > 0.82:
                    fileName = os.getcwd() + "/model_depository/" + self.model_type + "_" + str(test_acc * 100.0) + ".pt"
                    torch.save(self.model, fileName)
            print("epoch {:5d}, loss: {:.4f}, traain_acc: {:.4f}, test_acc: {:.4f}".format(i , loss, train_acc, test_acc));

        return epochs, train_accs, test_accs,best_acc,best_labels, best_preds


# +
def plot(net,epoch, train, test, pretrained, training_type):
#     plt.figure(figsize=(8, 6))
    
    plt.suptitle("Comparison of accrucy curve." + "(" + net + ")")
    plt.xlabel("epoch")
    plt.ylabel("Accuracy(%)")
    plt.plot(epoch, train, "r" if pretrained else "b")
    plt.plot(epoch, test, "g" if pretrained else "y")
    
    

def plot_confusion_matrix(best_preds, best_labels,model_type ,pretrained):
    plt.figure(figsize=(10, 8))
    print("Plotting confusion matrix...")

    
    ## Calculate confusion matrix
    cmatrix = confusion_matrix(best_labels,best_preds)
    p,r,f,s = precision_recall_fscore_support(best_labels, best_preds, average='macro')
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
    plt.savefig(os.getcwd() + "/experiment_result/confusion_matrix/cm_{}_{}_pretrained_{}_{}.png".format(model_type, pretrained_str, p * 100, r * 100))
    
    
def write_csv(net,pretrained,lr,test_acc,epoch):
    fw = open("./record.csv","a");
    fw.write("{:20s}, {:20s},{:7.4f},{:9.4f}, {:3d}\n".format(net,"pretrained" if pretrained else "non-pretrained" ,lr,test_acc,epoch))


# +
from prefetch_generator import BackgroundGenerator

class DataLoader_pro(DataLoader):    
    def __iter__(self):
        return BackgroundGenerator(super().__iter__());


# -



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
    print(train_data.shape)
    print(train_label.shape)

    train_dataset = RetinopathyLoader(os.getcwd() + "/data/", "train");
    test_dataset = RetinopathyLoader(os.getcwd() + "/data/", "test");
#     train_dataset = RetinopathyLoader("../data/", "train");
#     test_dataset = RetinopathyLoader("../data/", "test");

    train_loader_lp = DataLoader_pro(train_dataset,batch_size = 10, shuffle=True)
    test_loader_lp = DataLoader_pro(test_dataset,batch_size = 10, shuffle=False)
    
    
    train_loader_ft = DataLoader_pro(train_dataset,batch_size = 10, shuffle=True)
    test_loader_ft = DataLoader_pro(test_dataset,batch_size = 10, shuffle=False)

    
    plt.figure(figsize=(10, 8))
    model_pretrain = model_creator("50",True,device);
    model_pretrain = model_pretrain.to(device)
    
    
    lr = 2.5e-3
    epoch, train_acc, test_acc, best_acc,_,_ = model_pretrain.train_and_eval(train_loader_lp, test_loader_lp, 5, "linear-probe", lr * 2, 0);
    epoch, train_acc, test_acc, best_acc,best_labels1, best_preds1 = model_pretrain.train_and_eval(train_loader_ft, test_loader_ft, 10, "fine-tune", lr, 0);
    plot("ResNet50",epoch,train_acc,test_acc,True, "fine_tune");


    
    print("Training scratch model...")
    model = model_creator("50",False,device);
    model = model.to(device)
    lr = 4e-3
    epoch, train_acc, test_acc, best_acc,best_labels2, best_preds2 = model.train_and_eval(train_loader_ft, test_loader_ft, 10, "fine-tune", lr,1e-4);
    plot("ResNet50",epoch,train_acc,test_acc,False, "fine_tune");
    plt.legend(["pretrained_model's training", "pretrained_model's testing", "non-pretrained_model's training", "non-pretrained_model's testing"])
    fileName = os.getcwd() + "/experiment_result/accuracy_curve"
    timestr += time.strftime("%Y%m%d-%H%M%S")
    fileName = timestr + "_ResNet50";
    fileName += ".png"
    plt.savefig(fileName)


    plot_confusion_matrix(best_preds1,best_labels1,"ResNet50", True)
    plot_confusion_matrix(best_preds2,best_labels2,"ResNet50", False)
    del model_pretrain
    del model;

    plt.figure(figsize=(10, 8))
    
    model_pretrain = model_creator("18",True,device);
    model_pretrain = model_pretrain.to(device)
    lr = 2e-3
    epoch, train_acc, test_acc, best_acc,_,_ = model_pretrain.train_and_eval(train_loader_lp, test_loader_lp, 5, "linear-probe", lr * 2,0);
    epoch, train_acc, test_acc, best_acc,best_labels1, best_preds1 = model_pretrain.train_and_eval(train_loader_ft, test_loader_ft, 10, "fine-tune", lr,0);
    plot("ResNet18",epoch,train_acc,test_acc,True, "fine_tune");
    
    print("Training scratch model...")
    model = model_creator("18",False,device);
    model = model.to(device)
    lr = 4e-3
    epoch, train_acc, test_acc, best_acc,best_labels2, best_preds2 = model.train_and_eval(train_loader_ft, test_loader_ft, 10, "fine-tune", lr,1e-4);
    plot("ResNet18",epoch,train_acc,test_acc,False, "fine_tune");
    
    plt.legend(["pretrained_model's training", "pretrained_model's testing", "non-pretrained_model's training", "non-pretrained_model's testing"])
    timestr = time.strftime("%Y%m%d-%H%M%S")
    fileName = os.getcwd() + "/experiment_result/accuracy_curve/"
    fileName += timestr + "_ResNet18";
    fileName += ".png"
    plt.savefig(fileName)

    plot_confusion_matrix(best_preds1,best_labels1,"ResNet18", True)
    plot_confusion_matrix(best_preds2,best_labels2,"ResNet18", False)
